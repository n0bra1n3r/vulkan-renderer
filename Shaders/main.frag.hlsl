struct FragmentInput
{
    float4 sv_position : SV_Position;
    float4 tint : COLOR0;
    float2 texCoord : TEXCOORD0;
};

Texture2D<float4> texture : register(t2, space0);
SamplerState textureSampler : register(s2, space0);

float4 main(FragmentInput input) : SV_Target
{
    return input.tint * texture.Sample(textureSampler, input.texCoord);
}