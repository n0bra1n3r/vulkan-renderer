#include "common.fxh"

struct VertexInput
{
    float3 position : ATTRIB0;
    float3 normal : ATTRIB1;
    float2 texCoord : ATTRIB2;
    uint sv_instanceID : SV_InstanceID;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

StructuredBuffer<StorageBuffer> ssbo : register(t1, space0);

float3 rotateFloat3(float3 v, float4 q)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

VertexOutput main(VertexInput input)
{
    StorageBuffer instanceData = ssbo[input.sv_instanceID];
    VertexOutput output;
    float3 animatedPosition = rotateFloat3(input.position, ubo.rotation);
    float4 worldPosition = mul(instanceData.model, float4(animatedPosition, 1.0));
    float4 viewPosition = mul(ubo.view, worldPosition);
    float4 clipPosition = mul(ubo.proj, viewPosition);
    output.sv_position = clipPosition;
    output.colour = instanceData.colour;
    output.normalWS = normalize(rotateFloat3(input.normal, ubo.rotation));
    output.texCoord = input.texCoord;
    output.instanceID = input.sv_instanceID;
    return output;
}