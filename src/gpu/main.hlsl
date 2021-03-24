struct CSInput {
    uint width;
    uint height;
    float time;
};

[[vk::push_constant]] CSInput input;

RWTexture2D<float4> output : register(u0);

[numthreads(8, 8, 1)] void main(uint3 dispatchThreadId
                                : SV_DispatchThreadID) {
    uint2 id = dispatchThreadId.xy;
    if (id.x >= input.width || id.y >= input.height)
        return;
    output[id] = float4(float(id.x) / float(input.width), float(id.y) / float(input.height), cos(input.time * 2.0) * 0.5 + 0.5, 1);
}