struct FragmentInput
{
    float4 sv_position : SV_Position;
    float4 tint : COLOR0;
    float4 worldPos : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
};

static const float3 lightDir = float3(1.0, 0.0, 0.0);

// Small epsilon to avoid self-intersection
static const float EPSILON = 0.01;

Texture2D<float4> texture : register(t2, space0);
SamplerState textureSampler : register(s2, space0);

RaytracingAccelerationStructure accelerationStructure : register(t3, space0);;

bool in_shadow(float4 pos)
{
    // Build the shadow ray
    RayDesc ray;
    ray.Origin = pos.xyz;
    ray.Direction = normalize(lightDir);
    ray.TMin = EPSILON;
    ray.TMax = 1e4;

    // Ray flags
    uint rayFlags =
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    // Create the ray query object
    RayQuery<RAY_FLAG_NONE> rq;
    rq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, ray);

    // Traverse the ray
    while (rq.Proceed())
    {
        rq.CommitNonOpaqueTriangleHit();
    }

    // Check if we committed a triangle hit
    bool hit = (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    return hit;
}

float4 main(FragmentInput input) : SV_Target
{
    float4 baseColor = input.tint * texture.Sample(textureSampler, input.texCoord);
    float4 worldPos = input.worldPos;
    bool inShadow = in_shadow(worldPos);
    if (inShadow) {
        baseColor *= 0.2;
    }
    return baseColor;
}