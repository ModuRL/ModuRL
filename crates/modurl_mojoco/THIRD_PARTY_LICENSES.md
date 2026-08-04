# Third-party notices

`modurl_mojoco` itself is distributed under the MIT License. None of its
current direct or resolved transitive dependencies requires this project to be
distributed under the GPL, LGPL, or AGPL.

## MuJoCo-rs

- Crate: `mujoco-rs` 5.0.0+MuJoCo-3.9.0
- Upstream: <https://github.com/davidhozic/mujoco-rs>
- Upstream license expression: `MIT OR Apache-2.0`
- License selected for this distribution: MIT

The wrapper's raw FFI declarations describe the Apache-2.0-licensed MuJoCo C
API. Their presence does not impose a copyleft license on this crate.

## MuJoCo

- Native library: MuJoCo 3.9.0
- Upstream: <https://github.com/google-deepmind/mujoco>
- License: Apache License 2.0

Apache-2.0 is a permissive license, not a GNU copyleft license. When the native
MuJoCo library is redistributed with a compiled application, include the
`LICENSE` and `THIRD_PARTY_NOTICES.txt` files from the downloaded MuJoCo
archive. This crate's build script copies them into Cargo's profile output as
`mujoco-LICENSE.txt` and `mujoco-THIRD_PARTY_NOTICES.txt` (and into `deps/`)
to make compliant redistribution easier.

MuJoCo's notice file covers its bundled permissively licensed components,
including libccd, Collisions, TinyXML2, LodePNG, Qhull, GLAD, GLFW,
TinyObjLoader, MarchingCubeCpp, {fmt}, OpenGL Mathematics, miniz, and relevant
LLVM/Clang runtime portions. A reference to GPLv2 inside the LLVM exception
text describes license compatibility; it does not license MuJoCo or this
project under GPLv2.

## ModuRL and Candle

- `modurl` is distributed under MIT. Its current Cargo manifest omits the SPDX
  metadata field, but the dependency's repository `LICENSE` file contains the
  MIT License.
- `candle-core` and the Candle crates resolved through `modurl` use
  `MIT OR Apache-2.0`; this distribution selects MIT.
- Remaining resolved Rust dependencies use permissive licenses such as MIT,
  Apache-2.0, BSD, ISC, Zlib, Unicode-3.0, CDLA-Permissive-2.0, or compatible
  alternatives. `r-efi` advertises
  `MIT OR Apache-2.0 OR LGPL-2.1-or-later`; this distribution selects MIT, so
  its LGPL alternative does not apply.

Optional `cuda`/`cudnn` features can require separately installed NVIDIA
software under NVIDIA's own terms. Those proprietary runtime terms do not turn
this crate into GPL software.

## Gymnasium environment models

The XML models in `assets/` are adapted from Gymnasium, copyright Farama
Foundation contributors and originally developed in OpenAI Gym. Gymnasium is
distributed under the MIT License:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
