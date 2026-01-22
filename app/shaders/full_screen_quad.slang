import include.triangles;

struct Position
{
    float2 offset;
    uint32_t triangle_index;
}

struct Info
{
    Triangle *triangles;
    Position start_position;
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

    var position = info.start_position;

    let forward = float2(1.0, 0.0);
    let up = float2(0.0, 1.0);
    let direction = up * in.uv.y + forward * in.uv.x * info.aspect;

    walk(position, direction * 5.0);

    var color = float3(0.0, 0.0, 1.0);
    if (position.triangle_index != uint32_t.maxValue)
    {
        let triangle = info.triangles[position.triangle_index];
        let r = abs(position.offset.x) / max(abs(triangle.bx), abs(triangle.cx));
        let g = abs(position.offset.y) / abs(triangle.cy);
        color = float3(r, g, 0.0);
    }

    out.color = float4(color, 1.0);

    return out;
}

void walk(inout Position position, float2 move_offset)
{
    if (position.triangle_index == uint32_t.maxValue)
        return;

    var distance = length(move_offset);
    var direction = move_offset / distance;

    var incoming_edge = uint8_t.maxValue;
    for (var step = 0; step < 1000; step++)
    {
        let triangle = info.triangles[position.triangle_index];

        let a = float2(0.0, 0.0);
        let b = float2(triangle.bx, 0.0);
        let c = float2(triangle.cx, triangle.cy);

        let ab = normalize(b - a);
        let ac = normalize(c - a);
        let bc = normalize(c - b);

        var ab_perp = float2(-ab.y, ab.x);
        ab_perp *= sign(dot(ab_perp, c - a));
        var ac_perp = float2(-ac.y, ac.x);
        ac_perp *= sign(dot(ac_perp, b - a));
        var bc_perp = float2(-bc.y, bc.x);
        bc_perp *= sign(dot(bc_perp, a - b));

        let ab_dist = dot(a - position.offset, ab_perp) / dot(direction, ab_perp);
        let ac_dist = dot(a - position.offset, ac_perp) / dot(direction, ac_perp);
        let bc_dist = dot(b - position.offset, bc_perp) / dot(direction, bc_perp);

        var edge = uint8_t.maxValue;
        var smallest_distance_to_edge = float.maxValue;
        if (smallest_distance_to_edge > ab_dist && ab_dist >= 0 && incoming_edge != 0)
        {
            smallest_distance_to_edge = ab_dist;
            edge = 0;
        }
        if (smallest_distance_to_edge > ac_dist && ac_dist >= 0 && incoming_edge != 1)
        {
            smallest_distance_to_edge = ac_dist;
            edge = 1;
        }
        if (smallest_distance_to_edge > bc_dist && bc_dist >= 0 && incoming_edge != 2)
        {
            smallest_distance_to_edge = bc_dist;
            edge = 2;
        }

        if (smallest_distance_to_edge == float.maxValue)
        {
            position.triangle_index = uint32_t.maxValue;
            return;
        }
        if (smallest_distance_to_edge > distance)
        {
            position.offset += direction * distance;
            return;
        }

        distance -= smallest_distance_to_edge;

        let edge_position = position.offset + direction * smallest_distance_to_edge;
        var edge_percent : float;
        var direction_percent : float;
        var direction_percent_perp : float;
        switch (edge)
        {
        case 0:
            edge_percent = dot(ab, edge_position - a);
            direction_percent = dot(ab, direction);
            direction_percent_perp = -dot(ab_perp, direction);
            break;
        case 1:
            edge_percent = dot(ac, edge_position - a);
            direction_percent = dot(ac, direction);
            direction_percent_perp = -dot(ac_perp, direction);
            break;
        case 2:
            edge_percent = dot(bc, edge_position - b);
            direction_percent = dot(bc, direction);
            direction_percent_perp = -dot(bc_perp, direction);
            break;
        }

        position.triangle_index = triangle.edge_triangles[edge];
        if (position.triangle_index == uint32_t.maxValue)
            return;
        let other_edge = triangle.edge_indices[edge];
        let other_triangle = info.triangles[position.triangle_index];

        incoming_edge = other_edge;

        let other_a = float2(0.0, 0.0);
        let other_b = float2(other_triangle.bx, 0.0);
        let other_c = float2(other_triangle.cx, other_triangle.cy);

        let other_ab = normalize(other_b - other_a);
        let other_ac = normalize(other_c - other_a);
        let other_bc = normalize(other_c - other_b);

        var other_ab_perp = float2(-other_ab.y, other_ab.x);
        other_ab_perp *= sign(dot(other_ab_perp, other_c - other_a));
        var other_ac_perp = float2(-other_ac.y, other_ac.x);
        other_ac_perp *= sign(dot(other_ac_perp, other_b - other_a));
        var other_bc_perp = float2(-other_bc.y, other_bc.x);
        other_bc_perp *= sign(dot(other_bc_perp, other_a - other_b));

        switch (other_edge)
        {
        case 0:
            position.offset = other_a + other_ab * edge_percent;
            direction = direction_percent * other_ab + direction_percent_perp * other_ab_perp;
            break;
        case 1:
            position.offset = other_a + other_ac * edge_percent;
            direction = direction_percent * other_ac + direction_percent_perp * other_ac_perp;
            break;
        case 2:
            position.offset = other_b + other_bc * edge_percent;
            direction = direction_percent * other_bc + direction_percent_perp * other_bc_perp;
            break;
        }
    }
}
