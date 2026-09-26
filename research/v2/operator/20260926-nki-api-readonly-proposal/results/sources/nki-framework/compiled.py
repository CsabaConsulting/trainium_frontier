"""[INTERNAL] CompileKernel and derived classes for compilation on Neuron devices.

INTERNAL MODULE - Not part of public API. May change without notice.
"""

import hashlib
import os
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import Any, Callable

from nki.compiler.driver import _birsim_executor, compile_to_bir, run_from_bir
from nki.compiler.frontend import NKIFrontend, ParserFrontend, TracerFrontend
from nki.compiler.ncc_driver import CompileOptions, NirResult
from nki.compiler.target import resolve_target
from nki.framework.kernel import Kernel, _typenames
from nki.language._nki_base import NKIObject
from nki.utils import get_binary_env_var


def compile_kernel_to_nir(
    kernel_func: Callable,
    *,
    inputs: dict[str, Any],
    compile_opts: CompileOptions,
    frontend: NKIFrontend | None = None,
    enable_cache: bool = True,
) -> NirResult:
    """Public API: compile NKI kernel to NirResult.

    Framework integrations (JAX, TorchXLA, TorchEager) use this as
    the single compilation entry point. Call build_config() on the
    returned NirResult to get framework config (FrameworkConfig).
    """
    if frontend is None:
        if get_binary_env_var("NKI_EXPERIMENTAL_TRACER_FE"):
            frontend = TracerFrontend()
        else:
            frontend = ParserFrontend()

    return compile_to_bir(
        kernel_func,
        frontend=frontend,
        inputs=inputs,
        compile_opts=compile_opts,
        enable_cache=enable_cache,
    )


# ==================================================
# Framework detection
# ==================================================


@dataclass(frozen=True)
class CompileKernel(Kernel):
    """A kernel configured for compilation and execution on a Neuron device."""

    artifacts_dir: str = ""
    """Optional path to directory for storing compiler artifacts."""

    target: str | None = None
    """Target NeuronDevice architecture. By default, infer from environment."""

    _enable_backend_opt: bool = False
    """Enable experimental MLIR optimization pipeline."""

    _executor: Callable | None = None
    """Kernel executor. By default, uses Spike executor."""

    _frontend_cls: type = ParserFrontend
    """Frontend class to use for compilation."""

    _enable_simulation: bool = False
    """Run BIRSim instead of hardware execution."""

    _enable_device_dump: bool = False
    """Insert DevicePrint after every SBUF-writing ISA instruction (tracer only)."""

    def __post_init__(self):
        super().__post_init__()
        # Enable compilation caching unless explicitly disabled via env var.
        # The cache lives on the function object so it persists across
        # Kernel instances. Deleting the attr also disables caching.
        if not get_binary_env_var("NKI_DISABLE_COMPILE_CACHE") and not hasattr(
            self.func, "_nki_compile_cache"
        ):
            self.func._nki_compile_cache = {}

    def _compile_opts(self):
        opts = CompileOptions(
            target=resolve_target(self.func, self.target),
            lnc=self.lnc,
            artifacts_dir=self.artifacts_dir,
            output_path="kernel.neff",
            enable_simulation=self._enable_simulation,
            enable_device_dump=self._enable_device_dump,
            address_rotation=self.address_rotation,
            enable_barrier_checker=self.barrier_check,
        )

        if not self._enable_backend_opt:
            opts = opts.disable_backend_optimizations()

        return opts

    @staticmethod
    def _hashable_arg(value, name):
        """Convert a kernel argument to something hashable, for use in a cache key.

        Tensors are represented by (shape, dtype).
        Lists and dictionaries are converted to tuples.
        Scalars and other hashable types pass through unchanged.

        Returns a tuple (value, is_hashable).
        (If is_hashable is False, then 'value' will be None.)
        """
        if isinstance(value, (tuple, list)):
            parts = []
            for i, e in enumerate(value):
                h, is_hashable = CompileKernel._hashable_arg(e, f"{name}[{i}]")
                if not is_hashable:
                    return None, False
                parts.append(h)
            return tuple(parts), True
        if isinstance(value, dict):
            parts = []
            for k, v in sorted(value.items()):
                h, is_hashable = CompileKernel._hashable_arg(v, f"{name}[{k}]")
                if not is_hashable:
                    return None, False
                parts.append((k, h))
            return tuple(parts), True
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            shape = (
                tuple(value.shape) if hasattr(value.shape, "__iter__") else value.shape
            )
            return (shape, str(value.dtype)), True

        if isinstance(value, (NKIObject, Hashable)):
            # Assumes hashable types have suitable __repr__ defined.
            # We introduce a special case for NKIObjects, which are *ordinarily* Hashable, but might not be
            # if annotated with the @dataclass decorator. See test_compile_cache_key.py for more details.
            return value, True

        return None, False

    def _generate_cache_key(self, inputs, compile_opts):
        """Generate a cache key from kernel inputs and compilation options.

        Returns None if any argument is not hashable (caching is skipped).

        Includes argument shapes/dtypes, kernel config (LNC, schedule),
        and compilation options that affect output (target, mode, pipeline
        options, frontend class). Path-only fields (artifacts_dir,
        output_path) are excluded since they don't affect compiled output.
        """
        key_parts = []
        for name, v in inputs.items():
            h, is_hashable = self._hashable_arg(v, name)
            if not is_hashable:
                return None
            key_parts.append(h)
        key_parts.append(self.lnc)
        key_parts.append(self.schedule)
        key_parts.append(compile_opts.target)
        key_parts.append(compile_opts.nki_opt_pipeline_options)
        key_parts.append(self._frontend_cls.__name__)
        key_parts.append(self._enable_backend_opt)
        key_parts.append(compile_opts.address_rotation)
        return hashlib.sha256(repr(tuple(key_parts)).encode()).hexdigest()

    def _get_compile_cache(self):
        """Return the compilation cache, or None if caching is disabled.

        The cache is stored on ``self.func`` so it persists across Kernel
        instances wrapping the same function. If the attribute is absent,
        caching is disabled — we do not create it implicitly.
        """
        return getattr(self.func, "_nki_compile_cache", None)

    def _cached_compile_to_bir(self, *, frontend, inputs, compile_opts):
        """Compile with caching: skip recompilation when inputs match.

        Caching is skipped when: the cache is disabled (no attr on func),
        or any input argument is not hashable (cache key is None).
        """
        cache = self._get_compile_cache()
        cache_key = None
        if cache is not None:
            cache_key = self._generate_cache_key(inputs, compile_opts)
            if cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
        result = compile_kernel_to_nir(
            self,
            inputs=inputs,
            compile_opts=compile_opts,
            frontend=frontend,
        )
        if cache is not None and cache_key is not None:
            cache[cache_key] = result
        return result

    def _build_backend_config(self, nir, _compilation_result=None):
        """Deprecated: use nir.build_config() instead. Kept for TorchNeuronEager compat."""
        import warnings

        warnings.warn(
            "_build_backend_config is deprecated. Use nir.build_config() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return nir.build_config().backend_config

    def _compute_mode(self, args, kwargs):
        # Detect appropriate subclass based on argument types.
        arg_list = list(args) + list(kwargs.values())
        arg_types = _typenames(*arg_list)

        if arg_list and any(t.startswith(("jaxlib.", "jax.")) for t in arg_types):
            from nki.framework.jax import JaxKernel

            return JaxKernel

        if arg_list and any(t.startswith("torch.") for t in arg_types):
            from nki.framework._torch_xla import TorchXlaKernel

            return TorchXlaKernel

        return StandaloneKernel

    def __call__(self, *args, **kwargs):
        # Detect framework based on argument types
        cls = self._compute_mode(args, kwargs)
        return self._to_subclass(cls)(*args, **kwargs)


@dataclass(frozen=True)
class StandaloneKernel(CompileKernel):
    def __call__(self, *args, **kwargs):
        frontend = self._frontend_cls(enable_backend_opt=self._enable_backend_opt)

        inputs = self._bind_args(args, kwargs)
        compile_opts = self._compile_opts()

        def _compile_and_run(compile_opts):
            nir = compile_kernel_to_nir(
                self,
                inputs=inputs,
                compile_opts=compile_opts,
                frontend=frontend,
                enable_cache=False,
            )
            if self._executor is not None:
                executor = self._executor
            elif self._enable_simulation:
                executor = _birsim_executor
            else:
                executor = None
            executor_kwargs = {}
            if executor is not None:
                executor_kwargs["executor"] = executor
            return run_from_bir(nir, compile_opts, inputs, **executor_kwargs)

        if compile_opts.artifacts_dir:
            return _compile_and_run(compile_opts)
        else:
            with tempfile.TemporaryDirectory(prefix="nki_") as work_dir:
                return _compile_and_run(
                    replace(
                        compile_opts,
                        artifacts_dir=work_dir,
                        output_path=os.path.join(work_dir, "kernel.neff"),
                    )
                )
