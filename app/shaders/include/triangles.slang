struct Triangle
{
    // ax is 0
    // ay is 0
    float bx;
    // by is 0
    float cx;
    float cy;

    uint32_t _padding1;

    uint32_t edge_triangles[3];
    uint8_t edge_indices[3];

    uint8_t _padding2;
}
