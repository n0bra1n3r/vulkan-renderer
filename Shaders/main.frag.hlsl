struct FragmentInput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
    float2 texCoord : TEXCOORD0;
    uint instanceID : TEXCOORD1;
};

Texture2D<float4> textures[] : register(t2, space0);
SamplerState textureSampler : register(s2, space0);

float4 main(FragmentInput input) : SV_Target
{
    float4 baseColor = textures[NonUniformResourceIndex(input.instanceID)].Sample(textureSampler, input.texCoord);
    return float4(input.colour * baseColor.rgb, 1.0);
}