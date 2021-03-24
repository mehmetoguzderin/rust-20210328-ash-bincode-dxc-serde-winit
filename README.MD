# Rust Ash Bincode Dxc Serde Winit

Initial release: 2021 March 28

```sh
clang-format --style=webkit -i src/gpu/main.hlsl
dxc -T cs_6_6 -spirv -fspv-target-env=vulkan1.1 -Fo src/gpu/main.hlsl.spv src/gpu/main.hlsl
spirv-opt -O -o src/gpu/main.hlsl.spv src/gpu/main.hlsl.spv
```
