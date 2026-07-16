# Third-party notices

`modurl_ale` incorporates the following upstream work. Copyright notices remain in the vendored files.

## Arcade Learning Environment and Stella

- Upstream project: Arcade Learning Environment
- Native version: ALE 0.6.0, including Stella 2.4.2-derived emulator code
- Source bundled by `ale-sys` 0.1.2
- `ale-rs` repository revision: `f97eaab6188953ece1f53c4022cdf8dfeb33e5e9`
- `ale-sys` source tree: `47609808c37b8c595dc1fad6af4af135ad544659`
- Embedded ALE source revision: `674dab9f9678d4c178f9ba8b53d884b1cdf8e75a`
- License: GNU General Public License version 2; this combined package is distributed as `GPL-2.0-only`
- Notices: `native/ale/License.txt` and `native/ale/Copyright.txt`

The following local files were copied from or adapted from `ale-sys` 0.1.2:

- `src/bindings.rs`: generated Rust FFI declarations copied from `ale-sys` and updated for Rust 2024 `unsafe extern` syntax.
- `build.rs`: adapted from the `ale-sys` native CMake build script.
- `native/ale/`: the ALE/Stella source that was embedded in `ale-sys`, with the ROM test fixture removed.

This source is vendored directly rather than linked through `ale-sys` because that crate's crates.io MIT metadata does not describe the bundled GPL-covered ALE/Stella code. The original ALE/Stella copyright and license files are retained with the source.

## ale-rs Rust wrapper

- Upstream project: `trolleyman/ale-rs`
- Crate: `ale` 0.1.3
- Native bindings source crate: `ale-sys` 0.1.2
- License stated by the wrapper: MIT
- Copyright: Callum Tolley and contributors

Portions of the private Rust FFI wrapper were derived from this project and modified to remove every bundled-ROM API. The MIT permission notice follows:

In particular, `src/ale.rs` is a reduced private adaptation of the `ale` 0.1.3 wrapper. It retains only the methods used by `AtariGym` and delegates to the vendored FFI declarations.

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
