/*
OpenCL RandomForestClassifier
classifier_class_name = TableRowClassifier
feature_specification = Quality.AREA Quality.ASPECT_RATIO Quality.GAUSS_CURVATURE Quality.MEAN_CURVATURE Quality.SPHERE_FITTED_CURVATURE_25_PERCENT Quality.SPHERE_FITTED_CURVATURE_50_PERCENT Quality.SPHERE_FITTED_CURVATURE_10_PERCENT
num_ground_truth_dimensions = 1
num_classes = 2
num_features = 7
max_depth = 2
num_trees = 10
feature_importances = 0.0,0.0,0.1,0.2,0.2,0.3,0.2
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_in4_TYPE in4, IMAGE_in5_TYPE in5, IMAGE_in6_TYPE in6, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float i4 = READ_IMAGE(in4, sampler, POS_in4_INSTANCE(x,y,z,0)).x;
 float i5 = READ_IMAGE(in5, sampler, POS_in5_INSTANCE(x,y,z,0)).x;
 float i6 = READ_IMAGE(in6, sampler, POS_in6_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i3<-0.0003277884388808161){
 s0+=4.0;
} else {
 s1+=9.0;
}
if(i2<0.0009746598079800606){
 s0+=4.0;
} else {
 s1+=9.0;
}
if(i3<-0.00023905180569272488){
 s0+=2.0;
} else {
 s1+=11.0;
}
if(i5<0.0005270727560855448){
 s1+=11.0;
} else {
 s0+=2.0;
}
if(i6<0.003961039707064629){
 s1+=9.0;
} else {
 s0+=4.0;
}
if(i4<0.0014258392620831728){
 s1+=11.0;
} else {
 s0+=2.0;
}
if(i6<0.003989465534687042){
 s1+=7.0;
} else {
 s0+=6.0;
}
if(i5<0.0005224181222729385){
 s1+=7.0;
} else {
 s0+=6.0;
}
if(i4<0.0013561488594859838){
 s1+=7.0;
} else {
 s0+=6.0;
}
if(i5<0.0005270727560855448){
 s1+=10.0;
} else {
 s0+=3.0;
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
