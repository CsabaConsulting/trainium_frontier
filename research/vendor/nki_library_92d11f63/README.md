# Isolated NKI Library source subset

Upstream: https://github.com/aws-neuron/nki-library

Pinned commit: `92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e`

License: Apache2.0; see LICENSE and the copyright notices in each source file.

The66 upstream Python files are unmodified. `manifest.json` lists hashes and
the minimal static relative-import closure for attention forward/backward and
MLP. The generated root `__init__.py` intentionally omits upstream's bundled
package override mechanism. This subset is imported only as
`research.vendor.nki_library_92d11f63`, never as `nkilib` or `nkilib_src`.

The separate scratch ReLU2 adapter is documented in
`research/fresh-start-nki-library-audit-2026-09-20.md`. It affects only its
isolated MLP test process, not these files or the installed Neuron package.
