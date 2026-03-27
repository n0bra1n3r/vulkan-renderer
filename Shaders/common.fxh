struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
    float3 normalWS : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
    uint instanceID : TEXCOORD2;
};

struct UniformBuffer
{
    float4x4 view;
    float4x4 proj;
    float4 rotation;
    float4 nLightDir;
};

struct StorageBuffer
{
    float4x4 model;
    float3 colour;
};