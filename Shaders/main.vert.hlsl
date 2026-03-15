struct VertexInput
{
    float3 position : ATTRIB0;
    float2 texCoord : ATTRIB1;
};

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
    float2 texCoord : TEXCOORD0;
};

struct UniformBuffer
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

struct StorageBuffer
{
    float3 colour;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

StructuredBuffer<StorageBuffer> ssbo : register(t1, space0);

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.position, 1.0))));
    output.colour = ssbo[0].colour;
    output.texCoord = input.texCoord;
    return output;
}