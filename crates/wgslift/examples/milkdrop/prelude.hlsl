// Macros that MilkDrop 2 defines for every warp and comp shader.
// Sources: the preset authoring guide and the 2.0 user manual by Ryan Geiss,
// and the foo_vis_milk2 preset guide where the first two are silent.

// q1..q32 are the components of _qa.._qh, in x, y, z, w order.
#define q1 _qa.x
#define q2 _qa.y
#define q3 _qa.z
#define q4 _qa.w
#define q5 _qb.x
#define q6 _qb.y
#define q7 _qb.z
#define q8 _qb.w
#define q9 _qc.x
#define q10 _qc.y
#define q11 _qc.z
#define q12 _qc.w
#define q13 _qd.x
#define q14 _qd.y
#define q15 _qd.z
#define q16 _qd.w
#define q17 _qe.x
#define q18 _qe.y
#define q19 _qe.z
#define q20 _qe.w
#define q21 _qf.x
#define q22 _qf.y
#define q23 _qf.z
#define q24 _qf.w
#define q25 _qg.x
#define q26 _qg.y
#define q27 _qg.z
#define q28 _qg.w
#define q29 _qh.x
#define q30 _qh.y
#define q31 _qh.z
#define q32 _qh.w

// MilkDrop 2.0a and later accept the lowercase forms.
#define tex2d tex2D
#define tex3d tex3D

// M_PI_2 is two pi, not half of pi.
#define M_PI 3.14159265359
#define M_PI_2 6.28318530718
#define M_INV_PI_2 0.159154943091895

#define GetMain(uv) (tex2D(sampler_main, uv).xyz)
#define GetPixel(uv) (tex2D(sampler_main, uv).xyz)
#define lum(a) dot(a, float3(0.32, 0.49, 0.29))
