"�@
VHostIDLE"IDLE(
1     �@9fffff6k@A     �@Ifffff6k@a�g���?i�g���?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      B@9      B@A      B@I      B@az��F�1�?i������?�Unknown
^HostGatherV2"GatherV2(1      ;@9      ;@A      ;@I      ;@a��uL%�?iD�z/=i�?�Unknown
dHostDataset"Iterator::Model(1     �E@9     �E@A      9@I      9@a1T|FP��?i�͔p���?�Unknown
�HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      5@9      5@A      5@I      5@a\�I�Wr�?ir�9�W��?�Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      4@9      4@A      4@I      4@aN ���S?i��7�7�?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(1      4@9      4@A      3@I      3@a��`u��}?iW�l��s�?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      2@9      "@A      2@I      "@az��F�1|?i^9���?�Unknown
q	HostDataset"Iterator::Model::ParallelMap(1      2@9      2@A      2@I      2@az��F�1|?ieȇ�K��?�Unknown
o
Host_FusedMatMul"sequential/dense/Relu(1      2@9      2@A      2@I      2@az��F�1|?ilW
��?�Unknown
fHostGreaterEqual"GreaterEqual(1      0@9      0@A      0@I      0@a����y?i9���N�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      ,@9      ,@A      ,@I      ,@a�Ib���u?i�E%�z�?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      (@9      (@A      (@I      (@a��//��r?i'�_�A��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      &@9      &@A      &@I      &@a�^� �:q?i��`u���?�Unknown
�HostDataset"=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate(1      ,@9      ,@A      $@I      $@aN ���So?i�)��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      "@9      "@A      "@I      "@az��F�1l?i��K�<��?�Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a����i?in(5�L�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a����i?iU�j\0�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a����i?i<R-lI�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a�Ib���e?i����Y_�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a�Ib���e?i�!�Gu�?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a�Ib���e?iy��5��?�Unknown
jHostReadVariableOp"ReadVariableOp(1      @9      @A      @I      @a��//��b?iǨ�^��?�Unknown
�HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a��//��b?it�1Ͱ�?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a��//��b?i!;���?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aN ���S_?i1�B��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @aN ���S_?iA߶���?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aN ���S_?iQ������?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aN ���S_?ia��j@�?�Unknown
�HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aN ���S_?iq�TD��?�Unknown
VHostCast"Cast(1      @9      @A      @I      @a����Y?i���%r�?�Unknown
� HostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a����Y?iW�>�*�?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a����Y?i�X��7�?�Unknown
�"HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a����Y?i=#(�	D�?�Unknown
}#HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a����Y?i�휫�P�?�Unknown
�$HostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(1      1@9      1@A      @I      @a��//��R?i�����Y�?�Unknown
s%HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a��//��R?i\�}]c�?�Unknown
�&HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a��//��R?i2��f�l�?�Unknown
`'HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a��//��R?iM�O)v�?�Unknown
i(HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a��//��R?i��9��?�Unknown
�)HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a��//��R?i�|*"���?�Unknown
�*HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a��//��R?i�B[��?�Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a����I?i�y�����?�Unknown
|,HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a����I?i�޶���?�Unknown
u-HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a����I?i8Dq�&��?�Unknown
�.HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a����I?ir�+�j��?�Unknown
}/HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a����I?i�澮��?�Unknown
u0HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a����I?i�s����?�Unknown
u1HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a����I?i �Z�6��?�Unknown
�2HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a����I?iZ>�z��?�Unknown
}3HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a����I?i��ρ���?�Unknown
4HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a����I?i��r��?�Unknown
|5HostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a����I?inDcF��?�Unknown
�6HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a����I?iB��S���?�Unknown
�7HostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1      �?9      �?A      �?I      �?a����9?i�\L���?�Unknown
T8HostMul"Mul(1      �?9      �?A      �?I      �?a����9?i|8�D���?�Unknown
w9HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?a����9?ik=���?�Unknown
w:HostCast"%gradient_tape/mean_squared_error/Cast(1      �?9      �?A      �?I      �?a����9?i��s5��?�Unknown
u;HostMul"$gradient_tape/mean_squared_error/Mul(1      �?9      �?A      �?I      �?a����9?iS��-4��?�Unknown
<HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      �?9      �?A      �?I      �?a����9?i�.&V��?�Unknown
w=HostMul"&gradient_tape/mean_squared_error/mul_1(1      �?9      �?A      �?I      �?a����9?i�5�x��?�Unknown
}>HostRealDiv"(gradient_tape/mean_squared_error/truediv(1      �?9      �?A      �?I      �?a����9?i*h����?�Unknown
�?HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a����9?iǚE���?�Unknown
�@HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a����9?id͢���?�Unknown
�AHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a����9?i      �?�Unknown*�?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      B@9      B@A      B@I      B@a�P�B�
�?i�P�B�
�?�Unknown
^HostGatherV2"GatherV2(1      ;@9      ;@A      ;@I      ;@a����Ǐ�?i�&M�4i�?�Unknown
dHostDataset"Iterator::Model(1     �E@9     �E@A      9@I      9@a��QNG9�?ix��m���?�Unknown
�HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      5@9      5@A      5@I      5@aĈ#F��?i�������?�Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      4@9      4@A      4@I      4@avA�a�?i�-[�l��?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(1      4@9      4@A      3@I      3@a\cq��5�?iZ	h%��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      2@9      "@A      2@I      "@a�P�B�
�?i�]vA�?�Unknown
qHostDataset"Iterator::Model::ParallelMap(1      2@9      2@A      2@I      2@a�P�B�
�?i,������?�Unknown
o	Host_FusedMatMul"sequential/dense/Relu(1      2@9      2@A      2@I      2@a�P�B�
�?iA�a��?�Unknown
f
HostGreaterEqual"GreaterEqual(1      0@9      0@A      0@I      0@a@+���?i�������?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      ,@9      ,@A      ,@I      ,@a�a�]�?i2�<$��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      (@9      (@A      (@I      (@a����?i9�t���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      &@9      &@A      &@I      &@ax��m���?i*T�P��?�Unknown
�HostDataset"=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate(1      ,@9      ,@A      $@I      $@avA�a�?i�5�X\�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      "@9      "@A      "@I      "@a�P�B�
�?iK@+��?�Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a@+���?i�I�&M��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a@+���?i�R�K�/�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a@+���?iY\cq���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a�a�]�?i�d�yH�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a�a�]�?i�lٲe��?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a�a�]�?i�t��QN�?�Unknown
jHostReadVariableOp"ReadVariableOp(1      @9      @A      @I      @a����?i�{��?�Unknown
�HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a����?i�.�?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a����?i�'���?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @avA�a�?iʏ?~��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @avA�a�?i���VZ�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @avA�a�?iz��m���?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @avA�a�?iR�B�
�?�Unknown
�HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @avA�a�?i*����r�?�Unknown
VHostCast"Cast(1      @9      @A      @I      @a@+���?i׫W�^��?�Unknown
�HostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a@+���?i���.�?�Unknown
� HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a@+���?i1����R�?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a@+���?i޹s�Ν�?�Unknown
}"HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a@+���?i��'����?�Unknown
�#HostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(1      1@9      1@A      @I      @a���|?i�.� �?�Unknown
s$HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a���|?i��5�X�?�Unknown
�%HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a���|?i�<$��?�Unknown
`&HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a���|?i��C2��?�Unknown
i'HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a���|?i�J@+�?�Unknown
�(HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a���|?i��QNG9�?�Unknown
�)HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a���|?i�X\cq�?�Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a@+��r?ipٲe˖�?�Unknown
|+HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a@+��r?i��o3��?�Unknown
u,HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a@+��r?i�fx���?�Unknown
�-HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a@+��r?ir����?�Unknown
}.HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a@+��r?i���k,�?�Unknown
u/HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a@+��r?i�t��Q�?�Unknown
u0HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a@+��r?it�Ν;w�?�Unknown
�1HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a@+��r?i��(����?�Unknown
}2HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a@+��r?i 산��?�Unknown
3HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a@+��r?iv�ܹs��?�Unknown
|4HostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a@+��r?i��6���?�Unknown
�5HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a@+��r?i"��C2�?�Unknown
�6HostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1      �?9      �?A      �?I      �?a@+��b?iM�=��D�?�Unknown
T7HostMul"Mul(1      �?9      �?A      �?I      �?a@+��b?ix��իW�?�Unknown
w8HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?a@+��b?i����_j�?�Unknown
w9HostCast"%gradient_tape/mean_squared_error/Cast(1      �?9      �?A      �?I      �?a@+��b?i��D�}�?�Unknown
u:HostMul"$gradient_tape/mean_squared_error/Mul(1      �?9      �?A      �?I      �?a@+��b?i����Ǐ�?�Unknown
;HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      �?9      �?A      �?I      �?a@+��b?i$���{��?�Unknown
w<HostMul"&gradient_tape/mean_squared_error/mul_1(1      �?9      �?A      �?I      �?a@+��b?iO�K�/��?�Unknown
}=HostRealDiv"(gradient_tape/mean_squared_error/truediv(1      �?9      �?A      �?I      �?a@+��b?iz������?�Unknown
�>HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a@+��b?i�������?�Unknown
�?HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a@+��b?i��R�K��?�Unknown
�@HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a@+��b?i�������?�Unknown