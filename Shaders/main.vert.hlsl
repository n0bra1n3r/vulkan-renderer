struct VertexInput
{
    float2 position : ATTRIB0;
};

struct VertexOutput
{
    float4 sv_position : SV_Position;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.sv_position = float4(input.position, 0.0, 1.0);
    return output;
}