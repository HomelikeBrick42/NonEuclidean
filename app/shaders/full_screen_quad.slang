import include.triangles;

struct Info
{
    Triangle *triangles;
    float aspect;
}

[vk::push_constant]
Info info;

struct VertexOutput
{
    float4 clip_position : SV_Position;
    float2 uv;
}

[shader("vertex")]
VertexOutput vertex(uint vertex_index: SV_VertexID)
{
    var out : VertexOutput;

    let x = float((vertex_index >> 0) & 1);
    let y = float((vertex_index >> 1) & 1);
    out.uv = float2(x, y) * 2.0 - 1.0;

    out.clip_position = float4(out.uv, 0.0, 1.0);

    return out;
}

struct FragmentOutput
{
    float4 color : SV_Target;
}

[shader("fragment")]
FragmentOutput fragment(VertexOutput in)
{
    var out : FragmentOutput;

    let forward = float3(1.0, 0.0, 0.0);
    let up = float3(0.0, 1.0, 0.0);
    let right = float3(0.0, 0.0, 1.0);
    let direction = normalize(forward + up * in.uv.y + right * in.uv.x * info.aspect);
    out.color = float4(direction * 0.5 + 0.5, 1.0);

    return out;
}
