struct VertexInput
{
    float3 position : ATTRIB0;
    float2 texCoord : ATTRIB1;
    uint instanceID : SV_InstanceID;
};

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float4 tint : COLOR0;
    float2 texCoord : TEXCOORD0;
};

struct UniformBuffer
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

struct InstanceData
{
    float4x4 transform;
    float4 tint;
    int textureIndex;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

StructuredBuffer<InstanceData> ssbo : register(t1, space0);

VertexOutput main(VertexInput input)
{
    InstanceData instanceData = ssbo[input.instanceID];
    
    VertexOutput output;
    float4 localPos = mul(ubo.model, float4(input.position, 1.0));
    float4 worldPos = mul(instanceData.transform, localPos);
    float4 viewPos = mul(ubo.view, worldPos);
    output.sv_position = mul(ubo.proj, viewPos);
    output.tint = instanceData.tint;
    output.texCoord = input.texCoord;
    return output;
}