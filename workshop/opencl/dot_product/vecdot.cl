inline unsigned int rotate_left(unsigned int x, unsigned int n) {
    return  (x << n) | (x >> (32-n));
}
inline unsigned int encrypt(unsigned int m, unsigned int key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void vecdot(unsigned int key1, unsinged int key2, __global int *out) {
    int global_idx = get_global_id(0);
    out[global_idx] = encrypt(global_idx, key1) * encrypt(global_idx, key2);
}

