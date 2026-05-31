#include "common.fxh"

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv       : TexCoord;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

Texture2D<float4> albedo : register(t1, space0);
SamplerState albedoSampler : register(s1, space0);
Texture2D<float4> normal : register(t2, space0);
SamplerState normalSampler : register(s2, space0);
Texture2D<float4> position : register(t3, space0);
SamplerState positionSampler : register(s3, space0);
Texture2D<float4> shadowMap : register(t4, space0);
SamplerState shadowSampler : register(s4, space0);

float4 main(VSOutput input) : SV_Target
{
    float4 colour = albedo.Sample(albedoSampler, input.uv);
    float4 normalWS = normal.Sample(normalSampler, input.uv);
    float4 positionWS = position.Sample(positionSampler, input.uv);

    float4 lightViewPos = mul(ubo.lightView, positionWS);
    float4 lightClipPos = mul(ubo.lightProj, lightViewPos);

    float diffuse = saturate(dot(normalize(normalWS.xyz), ubo.nLightDir.xyz));
    float3 lightNDC = lightClipPos.xyz / lightClipPos.w;
    float2 lightUV = lightNDC.xy * 0.5f + 0.5f;
    float lightDepth = lightNDC.z;
    float shadowFactor = 1.0f;
    if (lightUV.x >= 0.0f && lightUV.x <= 1.0f && lightUV.y >= 0.0f && lightUV.y <= 1.0f)
    {
        float shadowDepth = shadowMap.Sample(shadowSampler, lightUV).r;
        shadowFactor = (lightDepth > shadowDepth + 0.005f) ? 0.5f : 1.0f;
    }
    colour.rgb *= diffuse * shadowFactor;

    return float4(colour.rgb, 1.0);
}
