/*
OpenCL RandomForestClassifier
classifier_class_name = TableRowClassifier
feature_specification = Quality.AREA Quality.ASPECT_RATIO Quality.GAUSS_CURVATURE Quality.MEAN_CURVATURE Quality.SPHERE_FITTED_CURVATURE_HECTA_VOXEL Quality.SPHERE_FITTED_CURVATURE_KILO_VOXEL
num_ground_truth_dimensions = 1
num_classes = 2
num_features = 6
max_depth = 2
num_trees = 10
feature_importances = 0.014774211871627363,0.045757080610021775,0.4964465603808598,0.10980786142320526,0.3332142857142857,0.0
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_in4_TYPE in4, IMAGE_in5_TYPE in5, IMAGE_out_TYPE out) {
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
 float s0=0;
 float s1=0;
if(i2<0.01253616251051426){
 s0+=13.0;
} else {
 if(i1<1.5743114948272705){
  s1+=16.0;
 } else {
  s0+=2.0;
 }
}
if(i2<0.013402309268712997){
 if(i2<0.007590430788695812){
  s0+=13.0;
 } else {
  s0+=2.0;
  s1+=1.0;
 }
} else {
 s1+=15.0;
}
if(i2<0.00794646143913269){
 s0+=21.0;
} else {
 s1+=10.0;
}
if(i4<0.00028659982490353286){
 s0+=17.0;
} else {
 s1+=14.0;
}
if(i3<9.934128320310265e-05){
 if(i1<1.3991124629974365){
  s0+=9.0;
 } else {
  s0+=3.0;
  s1+=4.0;
 }
} else {
 s1+=15.0;
}
if(i4<0.00029376597376540303){
 s0+=21.0;
} else {
 s1+=10.0;
}
if(i3<9.934128320310265e-05){
 if(i2<0.007590430788695812){
  s0+=16.0;
 } else {
  s0+=1.0;
  s1+=3.0;
 }
} else {
 if(i2<0.04185527563095093){
  s0+=1.0;
  s1+=7.0;
 } else {
  s0+=3.0;
 }
}
if(i4<0.00028545514214783907){
 s0+=14.0;
} else {
 s1+=17.0;
}
if(i2<0.006332993507385254){
 s0+=11.0;
} else {
 if(i4<0.00028545514214783907){
  s0+=3.0;
 } else {
  s1+=17.0;
 }
}
if(i0<26.780094146728516){
 s1+=3.0;
} else {
 if(i2<0.011249952018260956){
  s0+=16.0;
 } else {
  s0+=1.0;
  s1+=11.0;
 }
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
