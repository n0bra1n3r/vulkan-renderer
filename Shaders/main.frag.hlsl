#include "common.fxh"

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

Texture2D<float4> textures[] : register(t2, space0);
SamplerState textureSampler : register(s2, space0);

float4 main(VertexOutput input) : SV_Target
{
    float4 baseColor = textures[NonUniformResourceIndex(input.instanceID)].Sample(textureSampler, input.texCoord);
    float diffuse = saturate(dot(normalize(input.normalWS), ubo.nLightDir.xyz));
    baseColor.rgb *= diffuse;
    return float4(input.colour * baseColor.rgb, 1.0);
}