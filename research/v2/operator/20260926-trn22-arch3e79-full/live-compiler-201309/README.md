# Exact compiler capture

Use `exact-byte-bundle.json` for the authenticated source bytes. The remote
`module.mlir` is **binary MLIR bytecode**, not UTF-8 text. Its exact 20,777 bytes
have SHA-256 `97682f6d3f3bc59622a56cc557660dcb7df167b28f21c53473d866e336beddfd`.
The initial local file named `module.mlir` is a lossy diagnostic text view and
MUST NOT be used as original compiler input or for cache-key equality.

The exact binary was recovered in a second bounded read before cleanup; its
SHA matches the first read. The compiler log's exact first 16,384 bytes are
also in the bundle; the text view adds one newline. The four-file read bound
was respected, no recursive cache scan or on-host copy occurred, and no SDK
import, model, cache, logging or compiler settings changed.

No corresponding authenticated cold3e79 input/cache-key pair is established.
The debug directory suffix is not called a NEFF key. Operator provides raw
evidence only; graph role/cache identity remain for independent analysis.
