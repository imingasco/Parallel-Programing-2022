#define MAXN 2000
#define RIGHT      in[idx + 1]
#define LEFT       in[idx - 1]
#define UP         in[idx - MAXN]
#define DOWN       in[idx + MAXN]
#define UPPERLEFT  in[idx - MAXN - 1]
#define UPPERRIGHT in[idx - MAXN + 1]
#define LOWERLEFT  in[idx + MAXN - 1]
#define LOWERRIGHT in[idx + MAXN + 1]
 
__kernel void next_round(int N, __global int *in, __global int *out) {
    int idx = get_global_id(0);
    int r = idx / MAXN, c = idx % MAXN;
    int live;
 
    if (r >= N || c >= N)
        return;
 
    if (r == 0 && c == 0) {
        live = RIGHT + DOWN + LOWERRIGHT;
    } else if (r == 0 && c == N - 1) {
        live = LEFT + LOWERLEFT + DOWN;
    } else if (r == N - 1 && c == 0) {
        live = RIGHT + UP + UPPERRIGHT;
    } else if (r == N - 1 && c == N - 1) {
        live = LEFT + UP + UPPERLEFT;
    } else if (r == 0) {
        live = LEFT + RIGHT + LOWERLEFT + DOWN + LOWERRIGHT;
    } else if (c == 0) {
        live = UP + DOWN + UPPERRIGHT + RIGHT + LOWERRIGHT;
    } else if (r == N - 1) {
        live = LEFT + RIGHT + UPPERLEFT + UP + UPPERRIGHT;
    } else if (c == N - 1) {
        live = UP + DOWN + UPPERLEFT + LEFT + LOWERLEFT;
    } else {
        live = UPPERLEFT + UP + UPPERRIGHT + LEFT + RIGHT + LOWERLEFT + DOWN + LOWERRIGHT;
    }
    if (in[idx] == 1 && (live == 2 || live == 3)) {
        out[idx] = 1;
    } else if (in[idx] == 0 && live == 3) {
        out[idx] = 1;
    } else {
        out[idx] = 0;
    }
}
