// Declarations that MilkDrop 2 gives every warp and comp shader. Each one is
// added only when the shader does not declare the same name.

uniform float time;
uniform float fps;
uniform float frame;
uniform float progress;
uniform float bass;
uniform float mid;
uniform float treb;
uniform float vol;
uniform float bass_att;
uniform float mid_att;
uniform float treb_att;
uniform float vol_att;
uniform float4 aspect;
uniform float4 texsize;
uniform float4 rand_frame;
uniform float4 rand_preset;
uniform float4 roam_cos;
uniform float4 roam_sin;
uniform float4 slow_roam_cos;
uniform float4 slow_roam_sin;
uniform float4 _qa;
uniform float4 _qb;
uniform float4 _qc;
uniform float4 _qd;
uniform float4 _qe;
uniform float4 _qf;
uniform float4 _qg;
uniform float4 _qh;
uniform float blur1_min;
uniform float blur1_max;
uniform float blur2_min;
uniform float blur2_max;
uniform float blur3_min;
uniform float blur3_max;
uniform float4x3 rot_s1;
uniform float4x3 rot_s2;
uniform float4x3 rot_s3;
uniform float4x3 rot_s4;
uniform float4x3 rot_d1;
uniform float4x3 rot_d2;
uniform float4x3 rot_d3;
uniform float4x3 rot_d4;
uniform float4x3 rot_f1;
uniform float4x3 rot_f2;
uniform float4x3 rot_f3;
uniform float4x3 rot_f4;
uniform float4x3 rot_vf1;
uniform float4x3 rot_vf2;
uniform float4x3 rot_vf3;
uniform float4x3 rot_vf4;
uniform float4x3 rot_uf1;
uniform float4x3 rot_uf2;
uniform float4x3 rot_uf3;
uniform float4x3 rot_uf4;
uniform float4x3 rot_rand1;
uniform float4x3 rot_rand2;
uniform float4x3 rot_rand3;
uniform float4x3 rot_rand4;

// The canvas and the noise textures, with the filter and wrap prefixes.
sampler sampler_main;
sampler sampler_fw_main;
sampler sampler_fc_main;
sampler sampler_pw_main;
sampler sampler_pc_main;
sampler sampler_noise_lq;
sampler sampler_fw_noise_lq;
sampler sampler_fc_noise_lq;
sampler sampler_pw_noise_lq;
sampler sampler_pc_noise_lq;
sampler sampler_noise_lq_lite;
sampler sampler_fw_noise_lq_lite;
sampler sampler_fc_noise_lq_lite;
sampler sampler_pw_noise_lq_lite;
sampler sampler_pc_noise_lq_lite;
sampler sampler_noise_mq;
sampler sampler_fw_noise_mq;
sampler sampler_fc_noise_mq;
sampler sampler_pw_noise_mq;
sampler sampler_pc_noise_mq;
sampler sampler_noise_hq;
sampler sampler_fw_noise_hq;
sampler sampler_fc_noise_hq;
sampler sampler_pw_noise_hq;
sampler sampler_pc_noise_hq;
sampler sampler_noisevol_lq;
sampler sampler_fw_noisevol_lq;
sampler sampler_fc_noisevol_lq;
sampler sampler_pw_noisevol_lq;
sampler sampler_pc_noisevol_lq;
sampler sampler_noisevol_hq;
sampler sampler_fw_noisevol_hq;
sampler sampler_fc_noisevol_hq;
sampler sampler_pw_noisevol_hq;
sampler sampler_pc_noisevol_hq;
sampler sampler_blur1;
sampler sampler_blur2;
sampler sampler_blur3;

uniform float4 texsize_main;
uniform float4 texsize_noise_lq;
uniform float4 texsize_noise_lq_lite;
uniform float4 texsize_noise_mq;
uniform float4 texsize_noise_hq;
uniform float4 texsize_noisevol_lq;
uniform float4 texsize_noisevol_hq;
uniform float4 texsize_blur1;
uniform float4 texsize_blur2;
uniform float4 texsize_blur3;

// GetBlurN unpacks the blur texture, which stores the colour range [blurN_min, blurN_max] as [0, 1].
float3 GetBlur1(float2 uv)
{
    return tex2D(sampler_blur1, uv).xyz * (blur1_max - blur1_min) + blur1_min;
}

float3 GetBlur2(float2 uv)
{
    return tex2D(sampler_blur2, uv).xyz * (blur2_max - blur2_min) + blur2_min;
}

float3 GetBlur3(float2 uv)
{
    return tex2D(sampler_blur3, uv).xyz * (blur3_max - blur3_min) + blur3_min;
}
