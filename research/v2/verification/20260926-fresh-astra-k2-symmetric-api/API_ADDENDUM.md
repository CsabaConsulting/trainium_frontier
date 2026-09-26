# Versioned API evidence supplement

Retrieved from the official AWS Neuron v2.31.0 documentation on 2026-09-26 UTC.
This is primary-source documentation, not evidence of the installed compiler's behavior
or a source-review conclusion. Whole-page bytes have not been archived or authenticated.

## NkiTensor.ap

Source: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.NkiTensor.html
Rendered lines 1126–1156, especially the parameter descriptions.

The documented access pattern replaces a view's shape and strides. The ordinary
`offset` parameter accepts an integer: it composes an element displacement with
the view's existing storage base. The default retains that existing base. This
parameter is distinct from `scalar_offset`, a dynamic index, and `vector_offset`,
a per-partition index tensor. `indirect_dim` selects the dimension whose stride
scales either kind of indirect index. The return value is another view.

## dma_copy

Source: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.dma_copy.html
Rendered lines 1099–1131 and 1193–1200.

The documented software-DGE path supports dynamic gather/scatter using SBUF
indices. Dynamic addressing enables bounds checking; the skip option suppresses
an out-of-bounds transfer. Vector indirection indexes an HBM tensor using a
partition-by-one SBUF index tensor. Scalar and vector indirection are mutually
exclusive. The ordinary integer `offset` parameter is not named in that exclusion.
When source and destination types differ, the documented conversion passes
through FP32. These descriptions do not prove that any particular combination
will compile or execute correctly on the installed SDK.
