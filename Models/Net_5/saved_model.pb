??:
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??3
t
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable
m
Variable/Read/ReadVariableOpReadVariableOpVariable*&
_output_shapes
:*
dtype0
x

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_1
q
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*&
_output_shapes
:*
dtype0
x

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
q
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*&
_output_shapes
: *
dtype0
x

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
q
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*&
_output_shapes
: *
dtype0
x

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
q
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*&
_output_shapes
: *
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:(*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
?
&conv2d_fixed/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&conv2d_fixed/batch_normalization/gamma
?
:conv2d_fixed/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp&conv2d_fixed/batch_normalization/gamma*
_output_shapes
:*
dtype0
?
%conv2d_fixed/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%conv2d_fixed/batch_normalization/beta
?
9conv2d_fixed/batch_normalization/beta/Read/ReadVariableOpReadVariableOp%conv2d_fixed/batch_normalization/beta*
_output_shapes
:*
dtype0
?
conv2d_na/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_na/conv2d/kernel
?
+conv2d_na/conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d_na/conv2d/kernel*&
_output_shapes
:*
dtype0
?
conv2d_na/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv2d_na/conv2d/bias
{
)conv2d_na/conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d_na/conv2d/bias*
_output_shapes
:*
dtype0
?
%conv2d_na/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%conv2d_na/batch_normalization_1/gamma
?
9conv2d_na/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp%conv2d_na/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
?
$conv2d_na/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$conv2d_na/batch_normalization_1/beta
?
8conv2d_na/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp$conv2d_na/batch_normalization_1/beta*
_output_shapes
:*
dtype0
?
*conv2d_fixed_1/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*conv2d_fixed_1/batch_normalization_2/gamma
?
>conv2d_fixed_1/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp*conv2d_fixed_1/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
?
)conv2d_fixed_1/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)conv2d_fixed_1/batch_normalization_2/beta
?
=conv2d_fixed_1/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp)conv2d_fixed_1/batch_normalization_2/beta*
_output_shapes
:*
dtype0
?
conv2d_na_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameconv2d_na_1/conv2d_1/kernel
?
/conv2d_na_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_na_1/conv2d_1/kernel*&
_output_shapes
: *
dtype0
?
conv2d_na_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_na_1/conv2d_1/bias
?
-conv2d_na_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_na_1/conv2d_1/bias*
_output_shapes
: *
dtype0
?
'conv2d_na_1/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'conv2d_na_1/batch_normalization_3/gamma
?
;conv2d_na_1/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp'conv2d_na_1/batch_normalization_3/gamma*
_output_shapes
: *
dtype0
?
&conv2d_na_1/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&conv2d_na_1/batch_normalization_3/beta
?
:conv2d_na_1/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp&conv2d_na_1/batch_normalization_3/beta*
_output_shapes
: *
dtype0
?
*conv2d_fixed_2/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*conv2d_fixed_2/batch_normalization_4/gamma
?
>conv2d_fixed_2/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp*conv2d_fixed_2/batch_normalization_4/gamma*
_output_shapes
: *
dtype0
?
)conv2d_fixed_2/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)conv2d_fixed_2/batch_normalization_4/beta
?
=conv2d_fixed_2/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp)conv2d_fixed_2/batch_normalization_4/beta*
_output_shapes
: *
dtype0
?
conv2d_na_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:& *,
shared_nameconv2d_na_2/conv2d_2/kernel
?
/conv2d_na_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_na_2/conv2d_2/kernel*&
_output_shapes
:& *
dtype0
?
conv2d_na_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_na_2/conv2d_2/bias
?
-conv2d_na_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_na_2/conv2d_2/bias*
_output_shapes
: *
dtype0
?
'conv2d_na_2/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'conv2d_na_2/batch_normalization_5/gamma
?
;conv2d_na_2/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp'conv2d_na_2/batch_normalization_5/gamma*
_output_shapes
: *
dtype0
?
&conv2d_na_2/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&conv2d_na_2/batch_normalization_5/beta
?
:conv2d_na_2/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp&conv2d_na_2/batch_normalization_5/beta*
_output_shapes
: *
dtype0
?
conv2d_na_3/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*,
shared_nameconv2d_na_3/conv2d_3/kernel
?
/conv2d_na_3/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_na_3/conv2d_3/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_na_3/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv2d_na_3/conv2d_3/bias
?
-conv2d_na_3/conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_na_3/conv2d_3/bias*
_output_shapes
:@*
dtype0
?
'conv2d_na_3/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'conv2d_na_3/batch_normalization_6/gamma
?
;conv2d_na_3/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp'conv2d_na_3/batch_normalization_6/gamma*
_output_shapes
:@*
dtype0
?
&conv2d_na_3/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&conv2d_na_3/batch_normalization_6/beta
?
:conv2d_na_3/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp&conv2d_na_3/batch_normalization_6/beta*
_output_shapes
:@*
dtype0
?
conv2d_na_4/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *,
shared_nameconv2d_na_4/conv2d_4/kernel
?
/conv2d_na_4/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_na_4/conv2d_4/kernel*&
_output_shapes
:` *
dtype0
?
conv2d_na_4/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_na_4/conv2d_4/bias
?
-conv2d_na_4/conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_na_4/conv2d_4/bias*
_output_shapes
: *
dtype0
?
'conv2d_na_4/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'conv2d_na_4/batch_normalization_7/gamma
?
;conv2d_na_4/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp'conv2d_na_4/batch_normalization_7/gamma*
_output_shapes
: *
dtype0
?
&conv2d_na_4/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&conv2d_na_4/batch_normalization_7/beta
?
:conv2d_na_4/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp&conv2d_na_4/batch_normalization_7/beta*
_output_shapes
: *
dtype0
?
3conv2d_fixed__transpose/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53conv2d_fixed__transpose/batch_normalization_8/gamma
?
Gconv2d_fixed__transpose/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp3conv2d_fixed__transpose/batch_normalization_8/gamma*
_output_shapes
: *
dtype0
?
2conv2d_fixed__transpose/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42conv2d_fixed__transpose/batch_normalization_8/beta
?
Fconv2d_fixed__transpose/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp2conv2d_fixed__transpose/batch_normalization_8/beta*
_output_shapes
: *
dtype0
?
conv2d_na_5/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *,
shared_nameconv2d_na_5/conv2d_5/kernel
?
/conv2d_na_5/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_na_5/conv2d_5/kernel*&
_output_shapes
:@ *
dtype0
?
conv2d_na_5/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_na_5/conv2d_5/bias
?
-conv2d_na_5/conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_na_5/conv2d_5/bias*
_output_shapes
: *
dtype0
?
'conv2d_na_5/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'conv2d_na_5/batch_normalization_9/gamma
?
;conv2d_na_5/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp'conv2d_na_5/batch_normalization_9/gamma*
_output_shapes
: *
dtype0
?
&conv2d_na_5/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&conv2d_na_5/batch_normalization_9/beta
?
:conv2d_na_5/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp&conv2d_na_5/batch_normalization_9/beta*
_output_shapes
: *
dtype0
?
6conv2d_fixed__transpose_1/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86conv2d_fixed__transpose_1/batch_normalization_10/gamma
?
Jconv2d_fixed__transpose_1/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp6conv2d_fixed__transpose_1/batch_normalization_10/gamma*
_output_shapes
: *
dtype0
?
5conv2d_fixed__transpose_1/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75conv2d_fixed__transpose_1/batch_normalization_10/beta
?
Iconv2d_fixed__transpose_1/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp5conv2d_fixed__transpose_1/batch_normalization_10/beta*
_output_shapes
: *
dtype0
?
,conv2d_fixed/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,conv2d_fixed/batch_normalization/moving_mean
?
@conv2d_fixed/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp,conv2d_fixed/batch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
0conv2d_fixed/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20conv2d_fixed/batch_normalization/moving_variance
?
Dconv2d_fixed/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp0conv2d_fixed/batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
+conv2d_na/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+conv2d_na/batch_normalization_1/moving_mean
?
?conv2d_na/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp+conv2d_na/batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
/conv2d_na/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/conv2d_na/batch_normalization_1/moving_variance
?
Cconv2d_na/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp/conv2d_na/batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
?
0conv2d_fixed_1/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20conv2d_fixed_1/batch_normalization_2/moving_mean
?
Dconv2d_fixed_1/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp0conv2d_fixed_1/batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
?
4conv2d_fixed_1/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64conv2d_fixed_1/batch_normalization_2/moving_variance
?
Hconv2d_fixed_1/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp4conv2d_fixed_1/batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
?
-conv2d_na_1/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-conv2d_na_1/batch_normalization_3/moving_mean
?
Aconv2d_na_1/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp-conv2d_na_1/batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
?
1conv2d_na_1/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31conv2d_na_1/batch_normalization_3/moving_variance
?
Econv2d_na_1/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp1conv2d_na_1/batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
?
0conv2d_fixed_2/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20conv2d_fixed_2/batch_normalization_4/moving_mean
?
Dconv2d_fixed_2/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp0conv2d_fixed_2/batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
?
4conv2d_fixed_2/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64conv2d_fixed_2/batch_normalization_4/moving_variance
?
Hconv2d_fixed_2/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp4conv2d_fixed_2/batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
?
-conv2d_na_2/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-conv2d_na_2/batch_normalization_5/moving_mean
?
Aconv2d_na_2/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp-conv2d_na_2/batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
?
1conv2d_na_2/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31conv2d_na_2/batch_normalization_5/moving_variance
?
Econv2d_na_2/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp1conv2d_na_2/batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
?
-conv2d_na_3/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-conv2d_na_3/batch_normalization_6/moving_mean
?
Aconv2d_na_3/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp-conv2d_na_3/batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
?
1conv2d_na_3/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31conv2d_na_3/batch_normalization_6/moving_variance
?
Econv2d_na_3/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp1conv2d_na_3/batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
?
-conv2d_na_4/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-conv2d_na_4/batch_normalization_7/moving_mean
?
Aconv2d_na_4/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp-conv2d_na_4/batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
?
1conv2d_na_4/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31conv2d_na_4/batch_normalization_7/moving_variance
?
Econv2d_na_4/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp1conv2d_na_4/batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
?
9conv2d_fixed__transpose/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9conv2d_fixed__transpose/batch_normalization_8/moving_mean
?
Mconv2d_fixed__transpose/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp9conv2d_fixed__transpose/batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
?
=conv2d_fixed__transpose/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=conv2d_fixed__transpose/batch_normalization_8/moving_variance
?
Qconv2d_fixed__transpose/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp=conv2d_fixed__transpose/batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0
?
-conv2d_na_5/batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-conv2d_na_5/batch_normalization_9/moving_mean
?
Aconv2d_na_5/batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp-conv2d_na_5/batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
?
1conv2d_na_5/batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31conv2d_na_5/batch_normalization_9/moving_variance
?
Econv2d_na_5/batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp1conv2d_na_5/batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0
?
<conv2d_fixed__transpose_1/batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><conv2d_fixed__transpose_1/batch_normalization_10/moving_mean
?
Pconv2d_fixed__transpose_1/batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp<conv2d_fixed__transpose_1/batch_normalization_10/moving_mean*
_output_shapes
: *
dtype0
?
@conv2d_fixed__transpose_1/batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@conv2d_fixed__transpose_1/batch_normalization_10/moving_variance
?
Tconv2d_fixed__transpose_1/batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp@conv2d_fixed__transpose_1/batch_normalization_10/moving_variance*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
n
strides
w
bn
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
d
&conv
'bn
(regularization_losses
)trainable_variables
*	variables
+	keras_api
w
,strides
-pad
.w
/bn
0regularization_losses
1trainable_variables
2	variables
3	keras_api

4	keras_api
R
5regularization_losses
6trainable_variables
7	variables
8	keras_api
d
9conv
:bn
;regularization_losses
<trainable_variables
=	variables
>	keras_api
w
?strides
@pad
Aw
Bbn
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api

G	keras_api
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
d
Lconv
Mbn
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
d
Rconv
Sbn
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
R
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
d
\conv
]bn
^regularization_losses
_trainable_variables
`	variables
a	keras_api
?
bstrides
cpad
d	out_shape
ew
fbn
gregularization_losses
htrainable_variables
i	variables
j	keras_api
R
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
d
oconv
pbn
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
?
ustrides
vpad
w	out_shape
xw
ybn
zregularization_losses
{trainable_variables
|	variables
}	keras_api
T
~regularization_losses
trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?
?0
?1
2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
.13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
A24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
e47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
x58
?59
?60
?61
?62
?
regularization_losses
?layers
 ?layer_regularization_losses
trainable_variables
?layer_metrics
?metrics
	variables
?non_trainable_variables
 
 
OM
VARIABLE_VALUEVariable1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 

?0
?1
'
?0
?1
2
?3
?4
?
regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?metrics
 	variables
?non_trainable_variables
 
 
 
?
"regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
#trainable_variables
?metrics
$	variables
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
?0
?1
?2
?3
0
?0
?1
?2
?3
?4
?5
?
(regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
)trainable_variables
?metrics
*	variables
?non_trainable_variables
 
 
?0
?1
?2
?3
QO
VARIABLE_VALUE
Variable_11layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 

?0
?1
'
?0
?1
.2
?3
?4
?
0regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
1trainable_variables
?metrics
2	variables
?non_trainable_variables
 
 
 
 
?
5regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
6trainable_variables
?metrics
7	variables
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
?0
?1
?2
?3
0
?0
?1
?2
?3
?4
?5
?
;regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
<trainable_variables
?metrics
=	variables
?non_trainable_variables
 
 
?0
?1
?2
?3
QO
VARIABLE_VALUE
Variable_21layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUE
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 

?0
?1
'
?0
?1
A2
?3
?4
?
Cregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Dtrainable_variables
?metrics
E	variables
?non_trainable_variables
 
 
 
 
?
Hregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Itrainable_variables
?metrics
J	variables
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
?0
?1
?2
?3
0
?0
?1
?2
?3
?4
?5
?
Nregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Otrainable_variables
?metrics
P	variables
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
?0
?1
?2
?3
0
?0
?1
?2
?3
?4
?5
?
Tregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Utrainable_variables
?metrics
V	variables
?non_trainable_variables
 
 
 
?
Xregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Ytrainable_variables
?metrics
Z	variables
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
?0
?1
?2
?3
0
?0
?1
?2
?3
?4
?5
?
^regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
_trainable_variables
?metrics
`	variables
?non_trainable_variables
 
 
?0
?1
?2
?3
 
QO
VARIABLE_VALUE
Variable_31layer_with_weights-8/w/.ATTRIBUTES/VARIABLE_VALUE
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 

?0
?1
'
?0
?1
e2
?3
?4
?
gregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
htrainable_variables
?metrics
i	variables
?non_trainable_variables
 
 
 
?
kregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
ltrainable_variables
?metrics
m	variables
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
?0
?1
?2
?3
0
?0
?1
?2
?3
?4
?5
?
qregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
rtrainable_variables
?metrics
s	variables
?non_trainable_variables
 
 
?0
?1
?2
?3
 
RP
VARIABLE_VALUE
Variable_42layer_with_weights-10/w/.ATTRIBUTES/VARIABLE_VALUE
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 

?0
?1
'
?0
?1
x2
?3
?4
?
zregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
{trainable_variables
?metrics
|	variables
?non_trainable_variables
 
 
 
?
~regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?metrics
?	variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
lj
VARIABLE_VALUE&conv2d_fixed/batch_normalization/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%conv2d_fixed/batch_normalization/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv2d_na/conv2d/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_na/conv2d/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%conv2d_na/batch_normalization_1/gamma0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$conv2d_na/batch_normalization_1/beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE*conv2d_fixed_1/batch_normalization_2/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)conv2d_fixed_1/batch_normalization_2/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_na_1/conv2d_1/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_na_1/conv2d_1/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'conv2d_na_1/batch_normalization_3/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE&conv2d_na_1/batch_normalization_3/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*conv2d_fixed_2/batch_normalization_4/gamma1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE)conv2d_fixed_2/batch_normalization_4/beta1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_na_2/conv2d_2/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_na_2/conv2d_2/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'conv2d_na_2/batch_normalization_5/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE&conv2d_na_2/batch_normalization_5/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_na_3/conv2d_3/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_na_3/conv2d_3/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'conv2d_na_3/batch_normalization_6/gamma1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE&conv2d_na_3/batch_normalization_6/beta1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_na_4/conv2d_4/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_na_4/conv2d_4/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'conv2d_na_4/batch_normalization_7/gamma1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE&conv2d_na_4/batch_normalization_7/beta1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3conv2d_fixed__transpose/batch_normalization_8/gamma1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2conv2d_fixed__transpose/batch_normalization_8/beta1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_na_5/conv2d_5/kernel1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_na_5/conv2d_5/bias1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'conv2d_na_5/batch_normalization_9/gamma1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE&conv2d_na_5/batch_normalization_9/beta1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE6conv2d_fixed__transpose_1/batch_normalization_10/gamma1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5conv2d_fixed__transpose_1/batch_normalization_10/beta1trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE,conv2d_fixed/batch_normalization/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0conv2d_fixed/batch_normalization/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+conv2d_na/batch_normalization_1/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/conv2d_na/batch_normalization_1/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0conv2d_fixed_1/batch_normalization_2/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4conv2d_fixed_1/batch_normalization_2/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-conv2d_na_1/batch_normalization_3/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1conv2d_na_1/batch_normalization_3/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0conv2d_fixed_2/batch_normalization_4/moving_mean'variables/25/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4conv2d_fixed_2/batch_normalization_4/moving_variance'variables/26/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-conv2d_na_2/batch_normalization_5/moving_mean'variables/31/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1conv2d_na_2/batch_normalization_5/moving_variance'variables/32/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-conv2d_na_3/batch_normalization_6/moving_mean'variables/37/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1conv2d_na_3/batch_normalization_6/moving_variance'variables/38/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-conv2d_na_4/batch_normalization_7/moving_mean'variables/43/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1conv2d_na_4/batch_normalization_7/moving_variance'variables/44/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9conv2d_fixed__transpose/batch_normalization_8/moving_mean'variables/48/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=conv2d_fixed__transpose/batch_normalization_8/moving_variance'variables/49/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-conv2d_na_5/batch_normalization_9/moving_mean'variables/54/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1conv2d_na_5/batch_normalization_9/moving_variance'variables/55/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE<conv2d_fixed__transpose_1/batch_normalization_10/moving_mean'variables/59/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@conv2d_fixed__transpose_1/batch_normalization_10/moving_variance'variables/60/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
 
 
 
?
0
?1
?2
?3
?4
.5
?6
?7
?8
?9
A10
?11
?12
?13
?14
?15
?16
?17
?18
e19
?20
?21
?22
?23
x24
?25
?26
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

0
 
 
 

0
?1
?2
 
 
 
 
 
 

?0
?1

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

&0
'1
 
 
 

?0
?1
 
 
 
 
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

/0
 
 
 

.0
?1
?2
 
 
 
 
 
 

?0
?1

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

90
:1
 
 
 

?0
?1
 
 
 
 
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

B0
 
 
 

A0
?1
?2
 
 
 
 
 
 

?0
?1

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

L0
M1
 
 
 

?0
?1
 

?0
?1

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

R0
S1
 
 
 

?0
?1
 
 
 
 
 
 

?0
?1

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

\0
]1
 
 
 

?0
?1
 
 
 
 
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

f0
 
 
 

e0
?1
?2
 
 
 
 
 
 

?0
?1

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

o0
p1
 
 
 

?0
?1
 
 
 
 
 
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables

y0
 
 
 

x0
?1
?2
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 

?0
?1
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable&conv2d_fixed/batch_normalization/gamma%conv2d_fixed/batch_normalization/beta,conv2d_fixed/batch_normalization/moving_mean0conv2d_fixed/batch_normalization/moving_varianceconv2d_na/conv2d/kernelconv2d_na/conv2d/bias%conv2d_na/batch_normalization_1/gamma$conv2d_na/batch_normalization_1/beta+conv2d_na/batch_normalization_1/moving_mean/conv2d_na/batch_normalization_1/moving_variance
Variable_1*conv2d_fixed_1/batch_normalization_2/gamma)conv2d_fixed_1/batch_normalization_2/beta0conv2d_fixed_1/batch_normalization_2/moving_mean4conv2d_fixed_1/batch_normalization_2/moving_varianceconv2d_na_1/conv2d_1/kernelconv2d_na_1/conv2d_1/bias'conv2d_na_1/batch_normalization_3/gamma&conv2d_na_1/batch_normalization_3/beta-conv2d_na_1/batch_normalization_3/moving_mean1conv2d_na_1/batch_normalization_3/moving_variance
Variable_2*conv2d_fixed_2/batch_normalization_4/gamma)conv2d_fixed_2/batch_normalization_4/beta0conv2d_fixed_2/batch_normalization_4/moving_mean4conv2d_fixed_2/batch_normalization_4/moving_varianceconv2d_na_2/conv2d_2/kernelconv2d_na_2/conv2d_2/bias'conv2d_na_2/batch_normalization_5/gamma&conv2d_na_2/batch_normalization_5/beta-conv2d_na_2/batch_normalization_5/moving_mean1conv2d_na_2/batch_normalization_5/moving_varianceconv2d_na_3/conv2d_3/kernelconv2d_na_3/conv2d_3/bias'conv2d_na_3/batch_normalization_6/gamma&conv2d_na_3/batch_normalization_6/beta-conv2d_na_3/batch_normalization_6/moving_mean1conv2d_na_3/batch_normalization_6/moving_varianceconv2d_na_4/conv2d_4/kernelconv2d_na_4/conv2d_4/bias'conv2d_na_4/batch_normalization_7/gamma&conv2d_na_4/batch_normalization_7/beta-conv2d_na_4/batch_normalization_7/moving_mean1conv2d_na_4/batch_normalization_7/moving_variance
Variable_33conv2d_fixed__transpose/batch_normalization_8/gamma2conv2d_fixed__transpose/batch_normalization_8/beta9conv2d_fixed__transpose/batch_normalization_8/moving_mean=conv2d_fixed__transpose/batch_normalization_8/moving_varianceconv2d_na_5/conv2d_5/kernelconv2d_na_5/conv2d_5/bias'conv2d_na_5/batch_normalization_9/gamma&conv2d_na_5/batch_normalization_9/beta-conv2d_na_5/batch_normalization_9/moving_mean1conv2d_na_5/batch_normalization_9/moving_variance
Variable_46conv2d_fixed__transpose_1/batch_normalization_10/gamma5conv2d_fixed__transpose_1/batch_normalization_10/beta<conv2d_fixed__transpose_1/batch_normalization_10/moving_mean@conv2d_fixed__transpose_1/batch_normalization_10/moving_varianceconv2d_6/kernelconv2d_6/bias*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_4525
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
? 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_4/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp:conv2d_fixed/batch_normalization/gamma/Read/ReadVariableOp9conv2d_fixed/batch_normalization/beta/Read/ReadVariableOp+conv2d_na/conv2d/kernel/Read/ReadVariableOp)conv2d_na/conv2d/bias/Read/ReadVariableOp9conv2d_na/batch_normalization_1/gamma/Read/ReadVariableOp8conv2d_na/batch_normalization_1/beta/Read/ReadVariableOp>conv2d_fixed_1/batch_normalization_2/gamma/Read/ReadVariableOp=conv2d_fixed_1/batch_normalization_2/beta/Read/ReadVariableOp/conv2d_na_1/conv2d_1/kernel/Read/ReadVariableOp-conv2d_na_1/conv2d_1/bias/Read/ReadVariableOp;conv2d_na_1/batch_normalization_3/gamma/Read/ReadVariableOp:conv2d_na_1/batch_normalization_3/beta/Read/ReadVariableOp>conv2d_fixed_2/batch_normalization_4/gamma/Read/ReadVariableOp=conv2d_fixed_2/batch_normalization_4/beta/Read/ReadVariableOp/conv2d_na_2/conv2d_2/kernel/Read/ReadVariableOp-conv2d_na_2/conv2d_2/bias/Read/ReadVariableOp;conv2d_na_2/batch_normalization_5/gamma/Read/ReadVariableOp:conv2d_na_2/batch_normalization_5/beta/Read/ReadVariableOp/conv2d_na_3/conv2d_3/kernel/Read/ReadVariableOp-conv2d_na_3/conv2d_3/bias/Read/ReadVariableOp;conv2d_na_3/batch_normalization_6/gamma/Read/ReadVariableOp:conv2d_na_3/batch_normalization_6/beta/Read/ReadVariableOp/conv2d_na_4/conv2d_4/kernel/Read/ReadVariableOp-conv2d_na_4/conv2d_4/bias/Read/ReadVariableOp;conv2d_na_4/batch_normalization_7/gamma/Read/ReadVariableOp:conv2d_na_4/batch_normalization_7/beta/Read/ReadVariableOpGconv2d_fixed__transpose/batch_normalization_8/gamma/Read/ReadVariableOpFconv2d_fixed__transpose/batch_normalization_8/beta/Read/ReadVariableOp/conv2d_na_5/conv2d_5/kernel/Read/ReadVariableOp-conv2d_na_5/conv2d_5/bias/Read/ReadVariableOp;conv2d_na_5/batch_normalization_9/gamma/Read/ReadVariableOp:conv2d_na_5/batch_normalization_9/beta/Read/ReadVariableOpJconv2d_fixed__transpose_1/batch_normalization_10/gamma/Read/ReadVariableOpIconv2d_fixed__transpose_1/batch_normalization_10/beta/Read/ReadVariableOp@conv2d_fixed/batch_normalization/moving_mean/Read/ReadVariableOpDconv2d_fixed/batch_normalization/moving_variance/Read/ReadVariableOp?conv2d_na/batch_normalization_1/moving_mean/Read/ReadVariableOpCconv2d_na/batch_normalization_1/moving_variance/Read/ReadVariableOpDconv2d_fixed_1/batch_normalization_2/moving_mean/Read/ReadVariableOpHconv2d_fixed_1/batch_normalization_2/moving_variance/Read/ReadVariableOpAconv2d_na_1/batch_normalization_3/moving_mean/Read/ReadVariableOpEconv2d_na_1/batch_normalization_3/moving_variance/Read/ReadVariableOpDconv2d_fixed_2/batch_normalization_4/moving_mean/Read/ReadVariableOpHconv2d_fixed_2/batch_normalization_4/moving_variance/Read/ReadVariableOpAconv2d_na_2/batch_normalization_5/moving_mean/Read/ReadVariableOpEconv2d_na_2/batch_normalization_5/moving_variance/Read/ReadVariableOpAconv2d_na_3/batch_normalization_6/moving_mean/Read/ReadVariableOpEconv2d_na_3/batch_normalization_6/moving_variance/Read/ReadVariableOpAconv2d_na_4/batch_normalization_7/moving_mean/Read/ReadVariableOpEconv2d_na_4/batch_normalization_7/moving_variance/Read/ReadVariableOpMconv2d_fixed__transpose/batch_normalization_8/moving_mean/Read/ReadVariableOpQconv2d_fixed__transpose/batch_normalization_8/moving_variance/Read/ReadVariableOpAconv2d_na_5/batch_normalization_9/moving_mean/Read/ReadVariableOpEconv2d_na_5/batch_normalization_9/moving_variance/Read/ReadVariableOpPconv2d_fixed__transpose_1/batch_normalization_10/moving_mean/Read/ReadVariableOpTconv2d_fixed__transpose_1/batch_normalization_10/moving_variance/Read/ReadVariableOpConst*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference__traced_save_7576
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2
Variable_3
Variable_4conv2d_6/kernelconv2d_6/bias&conv2d_fixed/batch_normalization/gamma%conv2d_fixed/batch_normalization/betaconv2d_na/conv2d/kernelconv2d_na/conv2d/bias%conv2d_na/batch_normalization_1/gamma$conv2d_na/batch_normalization_1/beta*conv2d_fixed_1/batch_normalization_2/gamma)conv2d_fixed_1/batch_normalization_2/betaconv2d_na_1/conv2d_1/kernelconv2d_na_1/conv2d_1/bias'conv2d_na_1/batch_normalization_3/gamma&conv2d_na_1/batch_normalization_3/beta*conv2d_fixed_2/batch_normalization_4/gamma)conv2d_fixed_2/batch_normalization_4/betaconv2d_na_2/conv2d_2/kernelconv2d_na_2/conv2d_2/bias'conv2d_na_2/batch_normalization_5/gamma&conv2d_na_2/batch_normalization_5/betaconv2d_na_3/conv2d_3/kernelconv2d_na_3/conv2d_3/bias'conv2d_na_3/batch_normalization_6/gamma&conv2d_na_3/batch_normalization_6/betaconv2d_na_4/conv2d_4/kernelconv2d_na_4/conv2d_4/bias'conv2d_na_4/batch_normalization_7/gamma&conv2d_na_4/batch_normalization_7/beta3conv2d_fixed__transpose/batch_normalization_8/gamma2conv2d_fixed__transpose/batch_normalization_8/betaconv2d_na_5/conv2d_5/kernelconv2d_na_5/conv2d_5/bias'conv2d_na_5/batch_normalization_9/gamma&conv2d_na_5/batch_normalization_9/beta6conv2d_fixed__transpose_1/batch_normalization_10/gamma5conv2d_fixed__transpose_1/batch_normalization_10/beta,conv2d_fixed/batch_normalization/moving_mean0conv2d_fixed/batch_normalization/moving_variance+conv2d_na/batch_normalization_1/moving_mean/conv2d_na/batch_normalization_1/moving_variance0conv2d_fixed_1/batch_normalization_2/moving_mean4conv2d_fixed_1/batch_normalization_2/moving_variance-conv2d_na_1/batch_normalization_3/moving_mean1conv2d_na_1/batch_normalization_3/moving_variance0conv2d_fixed_2/batch_normalization_4/moving_mean4conv2d_fixed_2/batch_normalization_4/moving_variance-conv2d_na_2/batch_normalization_5/moving_mean1conv2d_na_2/batch_normalization_5/moving_variance-conv2d_na_3/batch_normalization_6/moving_mean1conv2d_na_3/batch_normalization_6/moving_variance-conv2d_na_4/batch_normalization_7/moving_mean1conv2d_na_4/batch_normalization_7/moving_variance9conv2d_fixed__transpose/batch_normalization_8/moving_mean=conv2d_fixed__transpose/batch_normalization_8/moving_variance-conv2d_na_5/batch_normalization_9/moving_mean1conv2d_na_5/batch_normalization_9/moving_variance<conv2d_fixed__transpose_1/batch_normalization_10/moving_mean@conv2d_fixed__transpose_1/batch_normalization_10/moving_variance*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_restore_7775??0
?
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_7245

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
X
,__inference_concatenate_5_layer_call_fn_6509
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_34142
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????(2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:??????????? :[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?6
?
C__inference_conv2d_na_layer_call_and_return_conditional_losses_2320

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
add/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
addY
ReluReluadd:z:0*
T0*1
_output_shapes
:???????????2
Relu?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:???????????::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_5453

inputs"
conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_5648

inputs"
conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z?:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?&
?
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_6444

inputs,
(conv2d_transpose_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity??%batch_normalization_10/AssignNewValue?'batch_normalization_10/AssignNewValue_1?6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   ?   @      2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:??????????? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_10/FusedBatchNormV3?
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue?
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1?
IdentityIdentity+batch_normalization_10/FusedBatchNormV3:y:0&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
+__inference_conv2d_fixed_layer_call_fn_5468

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_22262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1231

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_9_layer_call_fn_7289

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_3_layer_call_fn_6102

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_29142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
??
?#
__inference__traced_save_7576
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_4_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableopE
Asavev2_conv2d_fixed_batch_normalization_gamma_read_readvariableopD
@savev2_conv2d_fixed_batch_normalization_beta_read_readvariableop6
2savev2_conv2d_na_conv2d_kernel_read_readvariableop4
0savev2_conv2d_na_conv2d_bias_read_readvariableopD
@savev2_conv2d_na_batch_normalization_1_gamma_read_readvariableopC
?savev2_conv2d_na_batch_normalization_1_beta_read_readvariableopI
Esavev2_conv2d_fixed_1_batch_normalization_2_gamma_read_readvariableopH
Dsavev2_conv2d_fixed_1_batch_normalization_2_beta_read_readvariableop:
6savev2_conv2d_na_1_conv2d_1_kernel_read_readvariableop8
4savev2_conv2d_na_1_conv2d_1_bias_read_readvariableopF
Bsavev2_conv2d_na_1_batch_normalization_3_gamma_read_readvariableopE
Asavev2_conv2d_na_1_batch_normalization_3_beta_read_readvariableopI
Esavev2_conv2d_fixed_2_batch_normalization_4_gamma_read_readvariableopH
Dsavev2_conv2d_fixed_2_batch_normalization_4_beta_read_readvariableop:
6savev2_conv2d_na_2_conv2d_2_kernel_read_readvariableop8
4savev2_conv2d_na_2_conv2d_2_bias_read_readvariableopF
Bsavev2_conv2d_na_2_batch_normalization_5_gamma_read_readvariableopE
Asavev2_conv2d_na_2_batch_normalization_5_beta_read_readvariableop:
6savev2_conv2d_na_3_conv2d_3_kernel_read_readvariableop8
4savev2_conv2d_na_3_conv2d_3_bias_read_readvariableopF
Bsavev2_conv2d_na_3_batch_normalization_6_gamma_read_readvariableopE
Asavev2_conv2d_na_3_batch_normalization_6_beta_read_readvariableop:
6savev2_conv2d_na_4_conv2d_4_kernel_read_readvariableop8
4savev2_conv2d_na_4_conv2d_4_bias_read_readvariableopF
Bsavev2_conv2d_na_4_batch_normalization_7_gamma_read_readvariableopE
Asavev2_conv2d_na_4_batch_normalization_7_beta_read_readvariableopR
Nsavev2_conv2d_fixed__transpose_batch_normalization_8_gamma_read_readvariableopQ
Msavev2_conv2d_fixed__transpose_batch_normalization_8_beta_read_readvariableop:
6savev2_conv2d_na_5_conv2d_5_kernel_read_readvariableop8
4savev2_conv2d_na_5_conv2d_5_bias_read_readvariableopF
Bsavev2_conv2d_na_5_batch_normalization_9_gamma_read_readvariableopE
Asavev2_conv2d_na_5_batch_normalization_9_beta_read_readvariableopU
Qsavev2_conv2d_fixed__transpose_1_batch_normalization_10_gamma_read_readvariableopT
Psavev2_conv2d_fixed__transpose_1_batch_normalization_10_beta_read_readvariableopK
Gsavev2_conv2d_fixed_batch_normalization_moving_mean_read_readvariableopO
Ksavev2_conv2d_fixed_batch_normalization_moving_variance_read_readvariableopJ
Fsavev2_conv2d_na_batch_normalization_1_moving_mean_read_readvariableopN
Jsavev2_conv2d_na_batch_normalization_1_moving_variance_read_readvariableopO
Ksavev2_conv2d_fixed_1_batch_normalization_2_moving_mean_read_readvariableopS
Osavev2_conv2d_fixed_1_batch_normalization_2_moving_variance_read_readvariableopL
Hsavev2_conv2d_na_1_batch_normalization_3_moving_mean_read_readvariableopP
Lsavev2_conv2d_na_1_batch_normalization_3_moving_variance_read_readvariableopO
Ksavev2_conv2d_fixed_2_batch_normalization_4_moving_mean_read_readvariableopS
Osavev2_conv2d_fixed_2_batch_normalization_4_moving_variance_read_readvariableopL
Hsavev2_conv2d_na_2_batch_normalization_5_moving_mean_read_readvariableopP
Lsavev2_conv2d_na_2_batch_normalization_5_moving_variance_read_readvariableopL
Hsavev2_conv2d_na_3_batch_normalization_6_moving_mean_read_readvariableopP
Lsavev2_conv2d_na_3_batch_normalization_6_moving_variance_read_readvariableopL
Hsavev2_conv2d_na_4_batch_normalization_7_moving_mean_read_readvariableopP
Lsavev2_conv2d_na_4_batch_normalization_7_moving_variance_read_readvariableopX
Tsavev2_conv2d_fixed__transpose_batch_normalization_8_moving_mean_read_readvariableop\
Xsavev2_conv2d_fixed__transpose_batch_normalization_8_moving_variance_read_readvariableopL
Hsavev2_conv2d_na_5_batch_normalization_9_moving_mean_read_readvariableopP
Lsavev2_conv2d_na_5_batch_normalization_9_moving_variance_read_readvariableop[
Wsavev2_conv2d_fixed__transpose_1_batch_normalization_10_moving_mean_read_readvariableop_
[savev2_conv2d_fixed__transpose_1_batch_normalization_10_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-8/w/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-10/w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_4_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableopAsavev2_conv2d_fixed_batch_normalization_gamma_read_readvariableop@savev2_conv2d_fixed_batch_normalization_beta_read_readvariableop2savev2_conv2d_na_conv2d_kernel_read_readvariableop0savev2_conv2d_na_conv2d_bias_read_readvariableop@savev2_conv2d_na_batch_normalization_1_gamma_read_readvariableop?savev2_conv2d_na_batch_normalization_1_beta_read_readvariableopEsavev2_conv2d_fixed_1_batch_normalization_2_gamma_read_readvariableopDsavev2_conv2d_fixed_1_batch_normalization_2_beta_read_readvariableop6savev2_conv2d_na_1_conv2d_1_kernel_read_readvariableop4savev2_conv2d_na_1_conv2d_1_bias_read_readvariableopBsavev2_conv2d_na_1_batch_normalization_3_gamma_read_readvariableopAsavev2_conv2d_na_1_batch_normalization_3_beta_read_readvariableopEsavev2_conv2d_fixed_2_batch_normalization_4_gamma_read_readvariableopDsavev2_conv2d_fixed_2_batch_normalization_4_beta_read_readvariableop6savev2_conv2d_na_2_conv2d_2_kernel_read_readvariableop4savev2_conv2d_na_2_conv2d_2_bias_read_readvariableopBsavev2_conv2d_na_2_batch_normalization_5_gamma_read_readvariableopAsavev2_conv2d_na_2_batch_normalization_5_beta_read_readvariableop6savev2_conv2d_na_3_conv2d_3_kernel_read_readvariableop4savev2_conv2d_na_3_conv2d_3_bias_read_readvariableopBsavev2_conv2d_na_3_batch_normalization_6_gamma_read_readvariableopAsavev2_conv2d_na_3_batch_normalization_6_beta_read_readvariableop6savev2_conv2d_na_4_conv2d_4_kernel_read_readvariableop4savev2_conv2d_na_4_conv2d_4_bias_read_readvariableopBsavev2_conv2d_na_4_batch_normalization_7_gamma_read_readvariableopAsavev2_conv2d_na_4_batch_normalization_7_beta_read_readvariableopNsavev2_conv2d_fixed__transpose_batch_normalization_8_gamma_read_readvariableopMsavev2_conv2d_fixed__transpose_batch_normalization_8_beta_read_readvariableop6savev2_conv2d_na_5_conv2d_5_kernel_read_readvariableop4savev2_conv2d_na_5_conv2d_5_bias_read_readvariableopBsavev2_conv2d_na_5_batch_normalization_9_gamma_read_readvariableopAsavev2_conv2d_na_5_batch_normalization_9_beta_read_readvariableopQsavev2_conv2d_fixed__transpose_1_batch_normalization_10_gamma_read_readvariableopPsavev2_conv2d_fixed__transpose_1_batch_normalization_10_beta_read_readvariableopGsavev2_conv2d_fixed_batch_normalization_moving_mean_read_readvariableopKsavev2_conv2d_fixed_batch_normalization_moving_variance_read_readvariableopFsavev2_conv2d_na_batch_normalization_1_moving_mean_read_readvariableopJsavev2_conv2d_na_batch_normalization_1_moving_variance_read_readvariableopKsavev2_conv2d_fixed_1_batch_normalization_2_moving_mean_read_readvariableopOsavev2_conv2d_fixed_1_batch_normalization_2_moving_variance_read_readvariableopHsavev2_conv2d_na_1_batch_normalization_3_moving_mean_read_readvariableopLsavev2_conv2d_na_1_batch_normalization_3_moving_variance_read_readvariableopKsavev2_conv2d_fixed_2_batch_normalization_4_moving_mean_read_readvariableopOsavev2_conv2d_fixed_2_batch_normalization_4_moving_variance_read_readvariableopHsavev2_conv2d_na_2_batch_normalization_5_moving_mean_read_readvariableopLsavev2_conv2d_na_2_batch_normalization_5_moving_variance_read_readvariableopHsavev2_conv2d_na_3_batch_normalization_6_moving_mean_read_readvariableopLsavev2_conv2d_na_3_batch_normalization_6_moving_variance_read_readvariableopHsavev2_conv2d_na_4_batch_normalization_7_moving_mean_read_readvariableopLsavev2_conv2d_na_4_batch_normalization_7_moving_variance_read_readvariableopTsavev2_conv2d_fixed__transpose_batch_normalization_8_moving_mean_read_readvariableopXsavev2_conv2d_fixed__transpose_batch_normalization_8_moving_variance_read_readvariableopHsavev2_conv2d_na_5_batch_normalization_9_moving_mean_read_readvariableopLsavev2_conv2d_na_5_batch_normalization_9_moving_variance_read_readvariableopWsavev2_conv2d_fixed__transpose_1_batch_normalization_10_moving_mean_read_readvariableop[savev2_conv2d_fixed__transpose_1_batch_normalization_10_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : :(:::::::::: : : : : : :& : : : : @:@:@:@:` : : : : : :@ : : : : : ::::::: : : : : : :@:@: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
: :,(
&
_output_shapes
: :,(
&
_output_shapes
: :,(
&
_output_shapes
:(: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:& : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:` : 

_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:@ : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
:@: 7

_output_shapes
:@: 8

_output_shapes
: : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: : >

_output_shapes
: : ?

_output_shapes
: :@

_output_shapes
: 
?
o
E__inference_concatenate_layer_call_and_return_conditional_losses_2276

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_2205

inputs"
conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_3236

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_5/BiasAdd?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_9/FusedBatchNormV3?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_9/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?@::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?@
 
_user_specified_nameinputs
?
?
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_6269

inputs,
(conv2d_transpose_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   Z   ?       2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????Z? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
IdentityIdentity*batch_normalization_8/FusedBatchNormV3:y:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????-P :::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1928

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
q
E__inference_concatenate_layer_call_and_return_conditional_losses_5490
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
q
G__inference_concatenate_3_layer_call_and_return_conditional_losses_2970

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P`2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????-P`2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????-P :?????????-P@:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????-P@
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_3505
input_1
conv2d_fixed_2258
conv2d_fixed_2260
conv2d_fixed_2262
conv2d_fixed_2264
conv2d_fixed_2266
conv2d_na_2389
conv2d_na_2391
conv2d_na_2393
conv2d_na_2395
conv2d_na_2397
conv2d_na_2399
conv2d_fixed_1_2478
conv2d_fixed_1_2480
conv2d_fixed_1_2482
conv2d_fixed_1_2484
conv2d_fixed_1_2486
conv2d_na_1_2611
conv2d_na_1_2613
conv2d_na_1_2615
conv2d_na_1_2617
conv2d_na_1_2619
conv2d_na_1_2621
conv2d_fixed_2_2700
conv2d_fixed_2_2702
conv2d_fixed_2_2704
conv2d_fixed_2_2706
conv2d_fixed_2_2708
conv2d_na_2_2833
conv2d_na_2_2835
conv2d_na_2_2837
conv2d_na_2_2839
conv2d_na_2_2841
conv2d_na_2_2843
conv2d_na_3_2950
conv2d_na_3_2952
conv2d_na_3_2954
conv2d_na_3_2956
conv2d_na_3_2958
conv2d_na_3_2960
conv2d_na_4_3083
conv2d_na_4_3085
conv2d_na_4_3087
conv2d_na_4_3089
conv2d_na_4_3091
conv2d_na_4_3093 
conv2d_fixed__transpose_3174 
conv2d_fixed__transpose_3176 
conv2d_fixed__transpose_3178 
conv2d_fixed__transpose_3180 
conv2d_fixed__transpose_3182
conv2d_na_5_3305
conv2d_na_5_3307
conv2d_na_5_3309
conv2d_na_5_3311
conv2d_na_5_3313
conv2d_na_5_3315"
conv2d_fixed__transpose_1_3396"
conv2d_fixed__transpose_1_3398"
conv2d_fixed__transpose_1_3400"
conv2d_fixed__transpose_1_3402"
conv2d_fixed__transpose_1_3404
conv2d_6_3457
conv2d_6_3459
identity?? conv2d_6/StatefulPartitionedCall?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?$conv2d_fixed/StatefulPartitionedCall?&conv2d_fixed_1/StatefulPartitionedCall?&conv2d_fixed_2/StatefulPartitionedCall?/conv2d_fixed__transpose/StatefulPartitionedCall?1conv2d_fixed__transpose_1/StatefulPartitionedCall?!conv2d_na/StatefulPartitionedCall?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_1/StatefulPartitionedCall?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_2/StatefulPartitionedCall?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_3/StatefulPartitionedCall?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_4/StatefulPartitionedCall?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_5/StatefulPartitionedCall?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
$conv2d_fixed/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_fixed_2258conv2d_fixed_2260conv2d_fixed_2262conv2d_fixed_2264conv2d_fixed_2266*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_22052&
$conv2d_fixed/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall-conv2d_fixed/StatefulPartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_22762
concatenate/PartitionedCall?
!conv2d_na/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_na_2389conv2d_na_2391conv2d_na_2393conv2d_na_2395conv2d_na_2397conv2d_na_2399*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_na_layer_call_and_return_conditional_losses_23202#
!conv2d_na/StatefulPartitionedCall?
&conv2d_fixed_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0conv2d_fixed_1_2478conv2d_fixed_1_2480conv2d_fixed_1_2482conv2d_fixed_1_2484conv2d_fixed_1_2486*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_24252(
&conv2d_fixed_1/StatefulPartitionedCall?
tf.image.resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Z   ?   2
tf.image.resize/resize/size?
%tf.image.resize/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0$tf.image.resize/resize/size:output:0*
T0*0
_output_shapes
:?????????Z?*
half_pixel_centers(2'
%tf.image.resize/resize/ResizeBilinear?
concatenate_1/PartitionedCallPartitionedCall/conv2d_fixed_1/StatefulPartitionedCall:output:06tf.image.resize/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_24982
concatenate_1/PartitionedCall?
#conv2d_na_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_na_1_2611conv2d_na_1_2613conv2d_na_1_2615conv2d_na_1_2617conv2d_na_1_2619conv2d_na_1_2621*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_25422%
#conv2d_na_1/StatefulPartitionedCall?
&conv2d_fixed_2/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:0conv2d_fixed_2_2700conv2d_fixed_2_2702conv2d_fixed_2_2704conv2d_fixed_2_2706conv2d_fixed_2_2708*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_26472(
&conv2d_fixed_2/StatefulPartitionedCall?
tf.image.resize_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"-   P   2
tf.image.resize_1/resize/size?
'tf.image.resize_1/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0&tf.image.resize_1/resize/size:output:0*
T0*/
_output_shapes
:?????????-P*
half_pixel_centers(2)
'tf.image.resize_1/resize/ResizeBilinear?
concatenate_2/PartitionedCallPartitionedCall/conv2d_fixed_2/StatefulPartitionedCall:output:08tf.image.resize_1/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P&* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_27202
concatenate_2/PartitionedCall?
#conv2d_na_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_na_2_2833conv2d_na_2_2835conv2d_na_2_2837conv2d_na_2_2839conv2d_na_2_2841conv2d_na_2_2843*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_27642%
#conv2d_na_2/StatefulPartitionedCall?
#conv2d_na_3/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0conv2d_na_3_2950conv2d_na_3_2952conv2d_na_3_2954conv2d_na_3_2956conv2d_na_3_2958conv2d_na_3_2960*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_28812%
#conv2d_na_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0,conv2d_na_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_29702
concatenate_3/PartitionedCall?
#conv2d_na_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_na_4_3083conv2d_na_4_3085conv2d_na_4_3087conv2d_na_4_3089conv2d_na_4_3091conv2d_na_4_3093*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_30142%
#conv2d_na_4/StatefulPartitionedCall?
/conv2d_fixed__transpose/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_4/StatefulPartitionedCall:output:0conv2d_fixed__transpose_3174conv2d_fixed__transpose_3176conv2d_fixed__transpose_3178conv2d_fixed__transpose_3180conv2d_fixed__transpose_3182*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_312021
/conv2d_fixed__transpose/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:08conv2d_fixed__transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_31922
concatenate_4/PartitionedCall?
#conv2d_na_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_na_5_3305conv2d_na_5_3307conv2d_na_5_3309conv2d_na_5_3311conv2d_na_5_3313conv2d_na_5_3315*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_32362%
#conv2d_na_5/StatefulPartitionedCall?
1conv2d_fixed__transpose_1/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_5/StatefulPartitionedCall:output:0conv2d_fixed__transpose_1_3396conv2d_fixed__transpose_1_3398conv2d_fixed__transpose_1_3400conv2d_fixed__transpose_1_3402conv2d_fixed__transpose_1_3404*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_334223
1conv2d_fixed__transpose_1/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0:conv2d_fixed__transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_34142
concatenate_5/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_6_3457conv2d_6_3459*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_34462"
 conv2d_6/StatefulPartitionedCall?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_2389*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_1_2611*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_2_2833*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_3_2950*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_4_3083*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_5_3305*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_3457*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp%^conv2d_fixed/StatefulPartitionedCall'^conv2d_fixed_1/StatefulPartitionedCall'^conv2d_fixed_2/StatefulPartitionedCall0^conv2d_fixed__transpose/StatefulPartitionedCall2^conv2d_fixed__transpose_1/StatefulPartitionedCall"^conv2d_na/StatefulPartitionedCall:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_1/StatefulPartitionedCall>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_2/StatefulPartitionedCall>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_3/StatefulPartitionedCall>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_4/StatefulPartitionedCall>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_5/StatefulPartitionedCall>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2L
$conv2d_fixed/StatefulPartitionedCall$conv2d_fixed/StatefulPartitionedCall2P
&conv2d_fixed_1/StatefulPartitionedCall&conv2d_fixed_1/StatefulPartitionedCall2P
&conv2d_fixed_2/StatefulPartitionedCall&conv2d_fixed_2/StatefulPartitionedCall2b
/conv2d_fixed__transpose/StatefulPartitionedCall/conv2d_fixed__transpose/StatefulPartitionedCall2f
1conv2d_fixed__transpose_1/StatefulPartitionedCall1conv2d_fixed__transpose_1/StatefulPartitionedCall2F
!conv2d_na/StatefulPartitionedCall!conv2d_na/StatefulPartitionedCall2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_1/StatefulPartitionedCall#conv2d_na_1/StatefulPartitionedCall2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_2/StatefulPartitionedCall#conv2d_na_2/StatefulPartitionedCall2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_3/StatefulPartitionedCall#conv2d_na_3/StatefulPartitionedCall2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_4/StatefulPartitionedCall#conv2d_na_4/StatefulPartitionedCall2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_5/StatefulPartitionedCall#conv2d_na_5/StatefulPartitionedCall2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6666

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1647

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?!
?
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_5432

inputs"
conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
s
G__inference_concatenate_1_layer_call_and_return_conditional_losses_5685
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????Z?:?????????Z?:Z V
0
_output_shapes
:?????????Z?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????Z?
"
_user_specified_name
inputs/1
?
?
__inference_loss_fn_1_6703F
Bconv2d_na_conv2d_kernel_regularizer_square_readvariableop_resource
identity??9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_na_conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
IdentityIdentity+conv2d_na/conv2d/kernel/Regularizer/mul:z:0:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1543

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1335

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ِ
?*
 __inference__traced_restore_7775
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3!
assignvariableop_4_variable_4&
"assignvariableop_5_conv2d_6_kernel$
 assignvariableop_6_conv2d_6_bias=
9assignvariableop_7_conv2d_fixed_batch_normalization_gamma<
8assignvariableop_8_conv2d_fixed_batch_normalization_beta.
*assignvariableop_9_conv2d_na_conv2d_kernel-
)assignvariableop_10_conv2d_na_conv2d_bias=
9assignvariableop_11_conv2d_na_batch_normalization_1_gamma<
8assignvariableop_12_conv2d_na_batch_normalization_1_betaB
>assignvariableop_13_conv2d_fixed_1_batch_normalization_2_gammaA
=assignvariableop_14_conv2d_fixed_1_batch_normalization_2_beta3
/assignvariableop_15_conv2d_na_1_conv2d_1_kernel1
-assignvariableop_16_conv2d_na_1_conv2d_1_bias?
;assignvariableop_17_conv2d_na_1_batch_normalization_3_gamma>
:assignvariableop_18_conv2d_na_1_batch_normalization_3_betaB
>assignvariableop_19_conv2d_fixed_2_batch_normalization_4_gammaA
=assignvariableop_20_conv2d_fixed_2_batch_normalization_4_beta3
/assignvariableop_21_conv2d_na_2_conv2d_2_kernel1
-assignvariableop_22_conv2d_na_2_conv2d_2_bias?
;assignvariableop_23_conv2d_na_2_batch_normalization_5_gamma>
:assignvariableop_24_conv2d_na_2_batch_normalization_5_beta3
/assignvariableop_25_conv2d_na_3_conv2d_3_kernel1
-assignvariableop_26_conv2d_na_3_conv2d_3_bias?
;assignvariableop_27_conv2d_na_3_batch_normalization_6_gamma>
:assignvariableop_28_conv2d_na_3_batch_normalization_6_beta3
/assignvariableop_29_conv2d_na_4_conv2d_4_kernel1
-assignvariableop_30_conv2d_na_4_conv2d_4_bias?
;assignvariableop_31_conv2d_na_4_batch_normalization_7_gamma>
:assignvariableop_32_conv2d_na_4_batch_normalization_7_betaK
Gassignvariableop_33_conv2d_fixed__transpose_batch_normalization_8_gammaJ
Fassignvariableop_34_conv2d_fixed__transpose_batch_normalization_8_beta3
/assignvariableop_35_conv2d_na_5_conv2d_5_kernel1
-assignvariableop_36_conv2d_na_5_conv2d_5_bias?
;assignvariableop_37_conv2d_na_5_batch_normalization_9_gamma>
:assignvariableop_38_conv2d_na_5_batch_normalization_9_betaN
Jassignvariableop_39_conv2d_fixed__transpose_1_batch_normalization_10_gammaM
Iassignvariableop_40_conv2d_fixed__transpose_1_batch_normalization_10_betaD
@assignvariableop_41_conv2d_fixed_batch_normalization_moving_meanH
Dassignvariableop_42_conv2d_fixed_batch_normalization_moving_varianceC
?assignvariableop_43_conv2d_na_batch_normalization_1_moving_meanG
Cassignvariableop_44_conv2d_na_batch_normalization_1_moving_varianceH
Dassignvariableop_45_conv2d_fixed_1_batch_normalization_2_moving_meanL
Hassignvariableop_46_conv2d_fixed_1_batch_normalization_2_moving_varianceE
Aassignvariableop_47_conv2d_na_1_batch_normalization_3_moving_meanI
Eassignvariableop_48_conv2d_na_1_batch_normalization_3_moving_varianceH
Dassignvariableop_49_conv2d_fixed_2_batch_normalization_4_moving_meanL
Hassignvariableop_50_conv2d_fixed_2_batch_normalization_4_moving_varianceE
Aassignvariableop_51_conv2d_na_2_batch_normalization_5_moving_meanI
Eassignvariableop_52_conv2d_na_2_batch_normalization_5_moving_varianceE
Aassignvariableop_53_conv2d_na_3_batch_normalization_6_moving_meanI
Eassignvariableop_54_conv2d_na_3_batch_normalization_6_moving_varianceE
Aassignvariableop_55_conv2d_na_4_batch_normalization_7_moving_meanI
Eassignvariableop_56_conv2d_na_4_batch_normalization_7_moving_varianceQ
Massignvariableop_57_conv2d_fixed__transpose_batch_normalization_8_moving_meanU
Qassignvariableop_58_conv2d_fixed__transpose_batch_normalization_8_moving_varianceE
Aassignvariableop_59_conv2d_na_5_batch_normalization_9_moving_meanI
Eassignvariableop_60_conv2d_na_5_batch_normalization_9_moving_varianceT
Passignvariableop_61_conv2d_fixed__transpose_1_batch_normalization_10_moving_meanX
Tassignvariableop_62_conv2d_fixed__transpose_1_batch_normalization_10_moving_variance
identity_64??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-8/w/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-10/w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_4Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_6_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_6_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_conv2d_fixed_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_conv2d_fixed_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_conv2d_na_conv2d_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_conv2d_na_conv2d_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_conv2d_na_batch_normalization_1_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp8assignvariableop_12_conv2d_na_batch_normalization_1_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp>assignvariableop_13_conv2d_fixed_1_batch_normalization_2_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp=assignvariableop_14_conv2d_fixed_1_batch_normalization_2_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_conv2d_na_1_conv2d_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2d_na_1_conv2d_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp;assignvariableop_17_conv2d_na_1_batch_normalization_3_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp:assignvariableop_18_conv2d_na_1_batch_normalization_3_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp>assignvariableop_19_conv2d_fixed_2_batch_normalization_4_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp=assignvariableop_20_conv2d_fixed_2_batch_normalization_4_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_conv2d_na_2_conv2d_2_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_conv2d_na_2_conv2d_2_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp;assignvariableop_23_conv2d_na_2_batch_normalization_5_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp:assignvariableop_24_conv2d_na_2_batch_normalization_5_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp/assignvariableop_25_conv2d_na_3_conv2d_3_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp-assignvariableop_26_conv2d_na_3_conv2d_3_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp;assignvariableop_27_conv2d_na_3_batch_normalization_6_gammaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp:assignvariableop_28_conv2d_na_3_batch_normalization_6_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_conv2d_na_4_conv2d_4_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp-assignvariableop_30_conv2d_na_4_conv2d_4_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp;assignvariableop_31_conv2d_na_4_batch_normalization_7_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp:assignvariableop_32_conv2d_na_4_batch_normalization_7_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpGassignvariableop_33_conv2d_fixed__transpose_batch_normalization_8_gammaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpFassignvariableop_34_conv2d_fixed__transpose_batch_normalization_8_betaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_conv2d_na_5_conv2d_5_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp-assignvariableop_36_conv2d_na_5_conv2d_5_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp;assignvariableop_37_conv2d_na_5_batch_normalization_9_gammaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp:assignvariableop_38_conv2d_na_5_batch_normalization_9_betaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpJassignvariableop_39_conv2d_fixed__transpose_1_batch_normalization_10_gammaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpIassignvariableop_40_conv2d_fixed__transpose_1_batch_normalization_10_betaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp@assignvariableop_41_conv2d_fixed_batch_normalization_moving_meanIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpDassignvariableop_42_conv2d_fixed_batch_normalization_moving_varianceIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp?assignvariableop_43_conv2d_na_batch_normalization_1_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpCassignvariableop_44_conv2d_na_batch_normalization_1_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpDassignvariableop_45_conv2d_fixed_1_batch_normalization_2_moving_meanIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpHassignvariableop_46_conv2d_fixed_1_batch_normalization_2_moving_varianceIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpAassignvariableop_47_conv2d_na_1_batch_normalization_3_moving_meanIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpEassignvariableop_48_conv2d_na_1_batch_normalization_3_moving_varianceIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpDassignvariableop_49_conv2d_fixed_2_batch_normalization_4_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpHassignvariableop_50_conv2d_fixed_2_batch_normalization_4_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpAassignvariableop_51_conv2d_na_2_batch_normalization_5_moving_meanIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpEassignvariableop_52_conv2d_na_2_batch_normalization_5_moving_varianceIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpAassignvariableop_53_conv2d_na_3_batch_normalization_6_moving_meanIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpEassignvariableop_54_conv2d_na_3_batch_normalization_6_moving_varianceIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpAassignvariableop_55_conv2d_na_4_batch_normalization_7_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpEassignvariableop_56_conv2d_na_4_batch_normalization_7_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpMassignvariableop_57_conv2d_fixed__transpose_batch_normalization_8_moving_meanIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpQassignvariableop_58_conv2d_fixed__transpose_batch_normalization_8_moving_varianceIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpAassignvariableop_59_conv2d_na_5_batch_normalization_9_moving_meanIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpEassignvariableop_60_conv2d_na_5_batch_normalization_9_moving_varianceIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpPassignvariableop_61_conv2d_fixed__transpose_1_batch_normalization_10_moving_meanIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpTassignvariableop_62_conv2d_fixed__transpose_1_batch_normalization_10_moving_varianceIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63?
Identity_64IdentityIdentity_63:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_64"#
identity_64Identity_64:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
4__inference_batch_normalization_6_layer_call_fn_7063

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_17512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
q
G__inference_concatenate_1_layer_call_and_return_conditional_losses_2498

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????Z?:?????????Z?:X T
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs
?
?
-__inference_conv2d_fixed_1_layer_call_fn_5678

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_24462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6741

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?%
?
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_6247

inputs,
(conv2d_transpose_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   Z   ?       2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????Z? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1?
IdentityIdentity*batch_normalization_8/FusedBatchNormV3:y:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????-P :::::2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
?
(__inference_conv2d_na_layer_call_fn_5604

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_na_layer_call_and_return_conditional_losses_23532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:???????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_7_layer_call_fn_7144

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_5960

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_2/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_5/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P&::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P&
 
_user_specified_nameinputs
?
?
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_6466

inputs,
(conv2d_transpose_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity??6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   ?   @      2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:??????????? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3?
IdentityIdentity+batch_normalization_10/FusedBatchNormV3:y:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
(__inference_conv2d_na_layer_call_fn_5587

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_na_layer_call_and_return_conditional_losses_23532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:???????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_6538

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Maxm
subSubBiasAdd:output:0Max:output:0*
T0*1
_output_shapes
:???????????2
subV
ExpExpsub:z:0*
T0*1
_output_shapes
:???????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Sump
truedivRealDivExp:y:0Sum:output:0*
T0*1
_output_shapes
:???????????2	
truediv?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
IdentityIdentitytruediv:z:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????(
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5278

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*K
_read_only_resource_inputs-
+)	"#$%()*+./034569:;>?*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_38962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_5_layer_call_fn_6982

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_2797

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_2/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_5/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P&::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P&
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7175

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_3142

inputs,
(conv2d_transpose_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   Z   ?       2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????Z? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
IdentityIdentity*batch_normalization_8/FusedBatchNormV3:y:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????-P :::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_4_layer_call_fn_6223

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_30472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P`::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P`
 
_user_specified_nameinputs
?
?
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_2668

inputs"
conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp6^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_1:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_1_layer_call_fn_5782

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_25752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_3_layer_call_fn_6837

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_batch_normalization_layer_call_fn_6609

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_10962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_2446

inputs"
conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z?:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1751

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?F
__inference__wrapped_model_1034
input_15
1model_conv2d_fixed_conv2d_readvariableop_resourceB
>model_conv2d_fixed_batch_normalization_readvariableop_resourceD
@model_conv2d_fixed_batch_normalization_readvariableop_1_resourceS
Omodel_conv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_resourceU
Qmodel_conv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_1_resource9
5model_conv2d_na_conv2d_conv2d_readvariableop_resource:
6model_conv2d_na_conv2d_biasadd_readvariableop_resourceA
=model_conv2d_na_batch_normalization_1_readvariableop_resourceC
?model_conv2d_na_batch_normalization_1_readvariableop_1_resourceR
Nmodel_conv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceT
Pmodel_conv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3model_conv2d_fixed_1_conv2d_readvariableop_resourceF
Bmodel_conv2d_fixed_1_batch_normalization_2_readvariableop_resourceH
Dmodel_conv2d_fixed_1_batch_normalization_2_readvariableop_1_resourceW
Smodel_conv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceY
Umodel_conv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource=
9model_conv2d_na_1_conv2d_1_conv2d_readvariableop_resource>
:model_conv2d_na_1_conv2d_1_biasadd_readvariableop_resourceC
?model_conv2d_na_1_batch_normalization_3_readvariableop_resourceE
Amodel_conv2d_na_1_batch_normalization_3_readvariableop_1_resourceT
Pmodel_conv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceV
Rmodel_conv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3model_conv2d_fixed_2_conv2d_readvariableop_resourceF
Bmodel_conv2d_fixed_2_batch_normalization_4_readvariableop_resourceH
Dmodel_conv2d_fixed_2_batch_normalization_4_readvariableop_1_resourceW
Smodel_conv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceY
Umodel_conv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource=
9model_conv2d_na_2_conv2d_2_conv2d_readvariableop_resource>
:model_conv2d_na_2_conv2d_2_biasadd_readvariableop_resourceC
?model_conv2d_na_2_batch_normalization_5_readvariableop_resourceE
Amodel_conv2d_na_2_batch_normalization_5_readvariableop_1_resourceT
Pmodel_conv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceV
Rmodel_conv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource=
9model_conv2d_na_3_conv2d_3_conv2d_readvariableop_resource>
:model_conv2d_na_3_conv2d_3_biasadd_readvariableop_resourceC
?model_conv2d_na_3_batch_normalization_6_readvariableop_resourceE
Amodel_conv2d_na_3_batch_normalization_6_readvariableop_1_resourceT
Pmodel_conv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceV
Rmodel_conv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource=
9model_conv2d_na_4_conv2d_4_conv2d_readvariableop_resource>
:model_conv2d_na_4_conv2d_4_biasadd_readvariableop_resourceC
?model_conv2d_na_4_batch_normalization_7_readvariableop_resourceE
Amodel_conv2d_na_4_batch_normalization_7_readvariableop_1_resourceT
Pmodel_conv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceV
Rmodel_conv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceJ
Fmodel_conv2d_fixed__transpose_conv2d_transpose_readvariableop_resourceO
Kmodel_conv2d_fixed__transpose_batch_normalization_8_readvariableop_resourceQ
Mmodel_conv2d_fixed__transpose_batch_normalization_8_readvariableop_1_resource`
\model_conv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceb
^model_conv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource=
9model_conv2d_na_5_conv2d_5_conv2d_readvariableop_resource>
:model_conv2d_na_5_conv2d_5_biasadd_readvariableop_resourceC
?model_conv2d_na_5_batch_normalization_9_readvariableop_resourceE
Amodel_conv2d_na_5_batch_normalization_9_readvariableop_1_resourceT
Pmodel_conv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceV
Rmodel_conv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceL
Hmodel_conv2d_fixed__transpose_1_conv2d_transpose_readvariableop_resourceR
Nmodel_conv2d_fixed__transpose_1_batch_normalization_10_readvariableop_resourceT
Pmodel_conv2d_fixed__transpose_1_batch_normalization_10_readvariableop_1_resourcec
_model_conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resourcee
amodel_conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_6_conv2d_readvariableop_resource2
.model_conv2d_6_biasadd_readvariableop_resource
identity??%model/conv2d_6/BiasAdd/ReadVariableOp?$model/conv2d_6/Conv2D/ReadVariableOp?(model/conv2d_fixed/Conv2D/ReadVariableOp?Fmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp?Hmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?5model/conv2d_fixed/batch_normalization/ReadVariableOp?7model/conv2d_fixed/batch_normalization/ReadVariableOp_1?*model/conv2d_fixed_1/Conv2D/ReadVariableOp?Jmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Lmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?9model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp?;model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1?*model/conv2d_fixed_2/Conv2D/ReadVariableOp?Jmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Lmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?9model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp?;model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1?Smodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Umodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?Bmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?Dmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1?=model/conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp?Vmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Xmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?Emodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?Gmodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1??model/conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp?"model/conv2d_na/add/ReadVariableOp?Emodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Gmodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?4model/conv2d_na/batch_normalization_1/ReadVariableOp?6model/conv2d_na/batch_normalization_1/ReadVariableOp_1?-model/conv2d_na/conv2d/BiasAdd/ReadVariableOp?,model/conv2d_na/conv2d/Conv2D/ReadVariableOp?$model/conv2d_na_1/add/ReadVariableOp?Gmodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Imodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?6model/conv2d_na_1/batch_normalization_3/ReadVariableOp?8model/conv2d_na_1/batch_normalization_3/ReadVariableOp_1?1model/conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp?0model/conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp?$model/conv2d_na_2/add/ReadVariableOp?Gmodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Imodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?6model/conv2d_na_2/batch_normalization_5/ReadVariableOp?8model/conv2d_na_2/batch_normalization_5/ReadVariableOp_1?1model/conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp?0model/conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp?$model/conv2d_na_3/add/ReadVariableOp?Gmodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Imodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?6model/conv2d_na_3/batch_normalization_6/ReadVariableOp?8model/conv2d_na_3/batch_normalization_6/ReadVariableOp_1?1model/conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp?0model/conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp?$model/conv2d_na_4/add/ReadVariableOp?Gmodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Imodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?6model/conv2d_na_4/batch_normalization_7/ReadVariableOp?8model/conv2d_na_4/batch_normalization_7/ReadVariableOp_1?1model/conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp?0model/conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp?$model/conv2d_na_5/add/ReadVariableOp?Gmodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Imodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?6model/conv2d_na_5/batch_normalization_9/ReadVariableOp?8model/conv2d_na_5/batch_normalization_9/ReadVariableOp_1?1model/conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp?0model/conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp?
(model/conv2d_fixed/Conv2D/ReadVariableOpReadVariableOp1model_conv2d_fixed_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model/conv2d_fixed/Conv2D/ReadVariableOp?
model/conv2d_fixed/Conv2DConv2Dinput_10model/conv2d_fixed/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model/conv2d_fixed/Conv2D?
5model/conv2d_fixed/batch_normalization/ReadVariableOpReadVariableOp>model_conv2d_fixed_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype027
5model/conv2d_fixed/batch_normalization/ReadVariableOp?
7model/conv2d_fixed/batch_normalization/ReadVariableOp_1ReadVariableOp@model_conv2d_fixed_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype029
7model/conv2d_fixed/batch_normalization/ReadVariableOp_1?
Fmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_conv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02H
Fmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Hmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_conv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02J
Hmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
7model/conv2d_fixed/batch_normalization/FusedBatchNormV3FusedBatchNormV3"model/conv2d_fixed/Conv2D:output:0=model/conv2d_fixed/batch_normalization/ReadVariableOp:value:0?model/conv2d_fixed/batch_normalization/ReadVariableOp_1:value:0Nmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Pmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 29
7model/conv2d_fixed/batch_normalization/FusedBatchNormV3?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2;model/conv2d_fixed/batch_normalization/FusedBatchNormV3:y:0input_1&model/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
model/concatenate/concat?
,model/conv2d_na/conv2d/Conv2D/ReadVariableOpReadVariableOp5model_conv2d_na_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,model/conv2d_na/conv2d/Conv2D/ReadVariableOp?
model/conv2d_na/conv2d/Conv2DConv2D!model/concatenate/concat:output:04model/conv2d_na/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model/conv2d_na/conv2d/Conv2D?
-model/conv2d_na/conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_conv2d_na_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-model/conv2d_na/conv2d/BiasAdd/ReadVariableOp?
model/conv2d_na/conv2d/BiasAddBiasAdd&model/conv2d_na/conv2d/Conv2D:output:05model/conv2d_na/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
model/conv2d_na/conv2d/BiasAdd?
4model/conv2d_na/batch_normalization_1/ReadVariableOpReadVariableOp=model_conv2d_na_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype026
4model/conv2d_na/batch_normalization_1/ReadVariableOp?
6model/conv2d_na/batch_normalization_1/ReadVariableOp_1ReadVariableOp?model_conv2d_na_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype028
6model/conv2d_na/batch_normalization_1/ReadVariableOp_1?
Emodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_conv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02G
Emodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Gmodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_conv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Gmodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
6model/conv2d_na/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3'model/conv2d_na/conv2d/BiasAdd:output:0<model/conv2d_na/batch_normalization_1/ReadVariableOp:value:0>model/conv2d_na/batch_normalization_1/ReadVariableOp_1:value:0Mmodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Omodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 28
6model/conv2d_na/batch_normalization_1/FusedBatchNormV3?
"model/conv2d_na/add/ReadVariableOpReadVariableOp6model_conv2d_na_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/conv2d_na/add/ReadVariableOp?
model/conv2d_na/addAddV2:model/conv2d_na/batch_normalization_1/FusedBatchNormV3:y:0*model/conv2d_na/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/conv2d_na/add?
model/conv2d_na/ReluRelumodel/conv2d_na/add:z:0*
T0*1
_output_shapes
:???????????2
model/conv2d_na/Relu?
*model/conv2d_fixed_1/Conv2D/ReadVariableOpReadVariableOp3model_conv2d_fixed_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*model/conv2d_fixed_1/Conv2D/ReadVariableOp?
model/conv2d_fixed_1/Conv2DConv2D"model/conv2d_na/Relu:activations:02model/conv2d_fixed_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
model/conv2d_fixed_1/Conv2D?
9model/conv2d_fixed_1/batch_normalization_2/ReadVariableOpReadVariableOpBmodel_conv2d_fixed_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02;
9model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp?
;model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1ReadVariableOpDmodel_conv2d_fixed_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1?
Jmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpSmodel_conv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Lmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUmodel_conv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02N
Lmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
;model/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$model/conv2d_fixed_1/Conv2D:output:0Amodel/conv2d_fixed_1/batch_normalization_2/ReadVariableOp:value:0Cmodel/conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1:value:0Rmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Tmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z?:::::*
epsilon%o?:*
is_training( 2=
;model/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3?
!model/tf.image.resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Z   ?   2#
!model/tf.image.resize/resize/size?
+model/tf.image.resize/resize/ResizeBilinearResizeBilinear!model/concatenate/concat:output:0*model/tf.image.resize/resize/size:output:0*
T0*0
_output_shapes
:?????????Z?*
half_pixel_centers(2-
+model/tf.image.resize/resize/ResizeBilinear?
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_1/concat/axis?
model/concatenate_1/concatConcatV2?model/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3:y:0<model/tf.image.resize/resize/ResizeBilinear:resized_images:0(model/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?2
model/concatenate_1/concat?
0model/conv2d_na_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9model_conv2d_na_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0model/conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp?
!model/conv2d_na_1/conv2d_1/Conv2DConv2D#model/concatenate_1/concat:output:08model/conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2#
!model/conv2d_na_1/conv2d_1/Conv2D?
1model/conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:model_conv2d_na_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model/conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp?
"model/conv2d_na_1/conv2d_1/BiasAddBiasAdd*model/conv2d_na_1/conv2d_1/Conv2D:output:09model/conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2$
"model/conv2d_na_1/conv2d_1/BiasAdd?
6model/conv2d_na_1/batch_normalization_3/ReadVariableOpReadVariableOp?model_conv2d_na_1_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype028
6model/conv2d_na_1/batch_normalization_3/ReadVariableOp?
8model/conv2d_na_1/batch_normalization_3/ReadVariableOp_1ReadVariableOpAmodel_conv2d_na_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model/conv2d_na_1/batch_normalization_3/ReadVariableOp_1?
Gmodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_conv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02I
Gmodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Imodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_conv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Imodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
8model/conv2d_na_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+model/conv2d_na_1/conv2d_1/BiasAdd:output:0>model/conv2d_na_1/batch_normalization_3/ReadVariableOp:value:0@model/conv2d_na_1/batch_normalization_3/ReadVariableOp_1:value:0Omodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2:
8model/conv2d_na_1/batch_normalization_3/FusedBatchNormV3?
$model/conv2d_na_1/add/ReadVariableOpReadVariableOp:model_conv2d_na_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/conv2d_na_1/add/ReadVariableOp?
model/conv2d_na_1/addAddV2<model/conv2d_na_1/batch_normalization_3/FusedBatchNormV3:y:0,model/conv2d_na_1/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
model/conv2d_na_1/add?
model/conv2d_na_1/ReluRelumodel/conv2d_na_1/add:z:0*
T0*0
_output_shapes
:?????????Z? 2
model/conv2d_na_1/Relu?
*model/conv2d_fixed_2/Conv2D/ReadVariableOpReadVariableOp3model_conv2d_fixed_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*model/conv2d_fixed_2/Conv2D/ReadVariableOp?
model/conv2d_fixed_2/Conv2DConv2D$model/conv2d_na_1/Relu:activations:02model/conv2d_fixed_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
model/conv2d_fixed_2/Conv2D?
9model/conv2d_fixed_2/batch_normalization_4/ReadVariableOpReadVariableOpBmodel_conv2d_fixed_2_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02;
9model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp?
;model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1ReadVariableOpDmodel_conv2d_fixed_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1?
Jmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpSmodel_conv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Lmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUmodel_conv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02N
Lmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
;model/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$model/conv2d_fixed_2/Conv2D:output:0Amodel/conv2d_fixed_2/batch_normalization_4/ReadVariableOp:value:0Cmodel/conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1:value:0Rmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Tmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2=
;model/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3?
#model/tf.image.resize_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"-   P   2%
#model/tf.image.resize_1/resize/size?
-model/tf.image.resize_1/resize/ResizeBilinearResizeBilinear!model/concatenate/concat:output:0,model/tf.image.resize_1/resize/size:output:0*
T0*/
_output_shapes
:?????????-P*
half_pixel_centers(2/
-model/tf.image.resize_1/resize/ResizeBilinear?
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_2/concat/axis?
model/concatenate_2/concatConcatV2?model/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3:y:0>model/tf.image.resize_1/resize/ResizeBilinear:resized_images:0(model/concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P&2
model/concatenate_2/concat?
0model/conv2d_na_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp9model_conv2d_na_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype022
0model/conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp?
!model/conv2d_na_2/conv2d_2/Conv2DConv2D#model/concatenate_2/concat:output:08model/conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2#
!model/conv2d_na_2/conv2d_2/Conv2D?
1model/conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:model_conv2d_na_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model/conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp?
"model/conv2d_na_2/conv2d_2/BiasAddBiasAdd*model/conv2d_na_2/conv2d_2/Conv2D:output:09model/conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2$
"model/conv2d_na_2/conv2d_2/BiasAdd?
6model/conv2d_na_2/batch_normalization_5/ReadVariableOpReadVariableOp?model_conv2d_na_2_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype028
6model/conv2d_na_2/batch_normalization_5/ReadVariableOp?
8model/conv2d_na_2/batch_normalization_5/ReadVariableOp_1ReadVariableOpAmodel_conv2d_na_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model/conv2d_na_2/batch_normalization_5/ReadVariableOp_1?
Gmodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_conv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02I
Gmodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Imodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_conv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Imodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
8model/conv2d_na_2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+model/conv2d_na_2/conv2d_2/BiasAdd:output:0>model/conv2d_na_2/batch_normalization_5/ReadVariableOp:value:0@model/conv2d_na_2/batch_normalization_5/ReadVariableOp_1:value:0Omodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2:
8model/conv2d_na_2/batch_normalization_5/FusedBatchNormV3?
$model/conv2d_na_2/add/ReadVariableOpReadVariableOp:model_conv2d_na_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/conv2d_na_2/add/ReadVariableOp?
model/conv2d_na_2/addAddV2<model/conv2d_na_2/batch_normalization_5/FusedBatchNormV3:y:0,model/conv2d_na_2/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
model/conv2d_na_2/add?
model/conv2d_na_2/ReluRelumodel/conv2d_na_2/add:z:0*
T0*/
_output_shapes
:?????????-P 2
model/conv2d_na_2/Relu?
0model/conv2d_na_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp9model_conv2d_na_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype022
0model/conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp?
!model/conv2d_na_3/conv2d_3/Conv2DConv2D$model/conv2d_na_2/Relu:activations:08model/conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@*
paddingSAME*
strides
2#
!model/conv2d_na_3/conv2d_3/Conv2D?
1model/conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:model_conv2d_na_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp?
"model/conv2d_na_3/conv2d_3/BiasAddBiasAdd*model/conv2d_na_3/conv2d_3/Conv2D:output:09model/conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2$
"model/conv2d_na_3/conv2d_3/BiasAdd?
6model/conv2d_na_3/batch_normalization_6/ReadVariableOpReadVariableOp?model_conv2d_na_3_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/conv2d_na_3/batch_normalization_6/ReadVariableOp?
8model/conv2d_na_3/batch_normalization_6/ReadVariableOp_1ReadVariableOpAmodel_conv2d_na_3_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/conv2d_na_3/batch_normalization_6/ReadVariableOp_1?
Gmodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_conv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Imodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_conv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
8model/conv2d_na_3/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3+model/conv2d_na_3/conv2d_3/BiasAdd:output:0>model/conv2d_na_3/batch_normalization_6/ReadVariableOp:value:0@model/conv2d_na_3/batch_normalization_6/ReadVariableOp_1:value:0Omodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/conv2d_na_3/batch_normalization_6/FusedBatchNormV3?
$model/conv2d_na_3/add/ReadVariableOpReadVariableOp:model_conv2d_na_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/conv2d_na_3/add/ReadVariableOp?
model/conv2d_na_3/addAddV2<model/conv2d_na_3/batch_normalization_6/FusedBatchNormV3:y:0,model/conv2d_na_3/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
model/conv2d_na_3/add?
model/conv2d_na_3/ReluRelumodel/conv2d_na_3/add:z:0*
T0*/
_output_shapes
:?????????-P@2
model/conv2d_na_3/Relu?
model/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_3/concat/axis?
model/concatenate_3/concatConcatV2$model/conv2d_na_2/Relu:activations:0$model/conv2d_na_3/Relu:activations:0(model/concatenate_3/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P`2
model/concatenate_3/concat?
0model/conv2d_na_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp9model_conv2d_na_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype022
0model/conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp?
!model/conv2d_na_4/conv2d_4/Conv2DConv2D#model/concatenate_3/concat:output:08model/conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2#
!model/conv2d_na_4/conv2d_4/Conv2D?
1model/conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp:model_conv2d_na_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model/conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp?
"model/conv2d_na_4/conv2d_4/BiasAddBiasAdd*model/conv2d_na_4/conv2d_4/Conv2D:output:09model/conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2$
"model/conv2d_na_4/conv2d_4/BiasAdd?
6model/conv2d_na_4/batch_normalization_7/ReadVariableOpReadVariableOp?model_conv2d_na_4_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype028
6model/conv2d_na_4/batch_normalization_7/ReadVariableOp?
8model/conv2d_na_4/batch_normalization_7/ReadVariableOp_1ReadVariableOpAmodel_conv2d_na_4_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model/conv2d_na_4/batch_normalization_7/ReadVariableOp_1?
Gmodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_conv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02I
Gmodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Imodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_conv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Imodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
8model/conv2d_na_4/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3+model/conv2d_na_4/conv2d_4/BiasAdd:output:0>model/conv2d_na_4/batch_normalization_7/ReadVariableOp:value:0@model/conv2d_na_4/batch_normalization_7/ReadVariableOp_1:value:0Omodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2:
8model/conv2d_na_4/batch_normalization_7/FusedBatchNormV3?
$model/conv2d_na_4/add/ReadVariableOpReadVariableOp:model_conv2d_na_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/conv2d_na_4/add/ReadVariableOp?
model/conv2d_na_4/addAddV2<model/conv2d_na_4/batch_normalization_7/FusedBatchNormV3:y:0,model/conv2d_na_4/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
model/conv2d_na_4/add?
model/conv2d_na_4/ReluRelumodel/conv2d_na_4/add:z:0*
T0*/
_output_shapes
:?????????-P 2
model/conv2d_na_4/Relu?
:model/conv2d_fixed__transpose/conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   Z   ?       2<
:model/conv2d_fixed__transpose/conv2d_transpose/input_sizes?
=model/conv2d_fixed__transpose/conv2d_transpose/ReadVariableOpReadVariableOpFmodel_conv2d_fixed__transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02?
=model/conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp?
.model/conv2d_fixed__transpose/conv2d_transposeConv2DBackpropInputCmodel/conv2d_fixed__transpose/conv2d_transpose/input_sizes:output:0Emodel/conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp:value:0$model/conv2d_na_4/Relu:activations:0*
T0*0
_output_shapes
:?????????Z? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
20
.model/conv2d_fixed__transpose/conv2d_transpose?
Bmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOpReadVariableOpKmodel_conv2d_fixed__transpose_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?
Dmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1ReadVariableOpMmodel_conv2d_fixed__transpose_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1?
Smodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp\model_conv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02U
Smodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Umodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^model_conv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02W
Umodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
Dmodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3FusedBatchNormV37model/conv2d_fixed__transpose/conv2d_transpose:output:0Jmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp:value:0Lmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1:value:0[model/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0]model/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2F
Dmodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3?
model/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_4/concat/axis?
model/concatenate_4/concatConcatV2$model/conv2d_na_1/Relu:activations:0Hmodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3:y:0(model/concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?@2
model/concatenate_4/concat?
0model/conv2d_na_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp9model_conv2d_na_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype022
0model/conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp?
!model/conv2d_na_5/conv2d_5/Conv2DConv2D#model/concatenate_4/concat:output:08model/conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2#
!model/conv2d_na_5/conv2d_5/Conv2D?
1model/conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp:model_conv2d_na_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model/conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp?
"model/conv2d_na_5/conv2d_5/BiasAddBiasAdd*model/conv2d_na_5/conv2d_5/Conv2D:output:09model/conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2$
"model/conv2d_na_5/conv2d_5/BiasAdd?
6model/conv2d_na_5/batch_normalization_9/ReadVariableOpReadVariableOp?model_conv2d_na_5_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype028
6model/conv2d_na_5/batch_normalization_9/ReadVariableOp?
8model/conv2d_na_5/batch_normalization_9/ReadVariableOp_1ReadVariableOpAmodel_conv2d_na_5_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model/conv2d_na_5/batch_normalization_9/ReadVariableOp_1?
Gmodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_conv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02I
Gmodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Imodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_conv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Imodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
8model/conv2d_na_5/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3+model/conv2d_na_5/conv2d_5/BiasAdd:output:0>model/conv2d_na_5/batch_normalization_9/ReadVariableOp:value:0@model/conv2d_na_5/batch_normalization_9/ReadVariableOp_1:value:0Omodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2:
8model/conv2d_na_5/batch_normalization_9/FusedBatchNormV3?
$model/conv2d_na_5/add/ReadVariableOpReadVariableOp:model_conv2d_na_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/conv2d_na_5/add/ReadVariableOp?
model/conv2d_na_5/addAddV2<model/conv2d_na_5/batch_normalization_9/FusedBatchNormV3:y:0,model/conv2d_na_5/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
model/conv2d_na_5/add?
model/conv2d_na_5/ReluRelumodel/conv2d_na_5/add:z:0*
T0*0
_output_shapes
:?????????Z? 2
model/conv2d_na_5/Relu?
<model/conv2d_fixed__transpose_1/conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   ?   @      2>
<model/conv2d_fixed__transpose_1/conv2d_transpose/input_sizes?
?model/conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHmodel_conv2d_fixed__transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?model/conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp?
0model/conv2d_fixed__transpose_1/conv2d_transposeConv2DBackpropInputEmodel/conv2d_fixed__transpose_1/conv2d_transpose/input_sizes:output:0Gmodel/conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp:value:0$model/conv2d_na_5/Relu:activations:0*
T0*1
_output_shapes
:??????????? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
22
0model/conv2d_fixed__transpose_1/conv2d_transpose?
Emodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOpReadVariableOpNmodel_conv2d_fixed__transpose_1_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?
Gmodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1ReadVariableOpPmodel_conv2d_fixed__transpose_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gmodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1?
Vmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp_model_conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
Xmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpamodel_conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02Z
Xmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
Gmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV39model/conv2d_fixed__transpose_1/conv2d_transpose:output:0Mmodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp:value:0Omodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1:value:0^model/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0`model/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2I
Gmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3?
model/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_5/concat/axis?
model/concatenate_5/concatConcatV2"model/conv2d_na/Relu:activations:0Kmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3:y:0(model/concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????(2
model/concatenate_5/concat?
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02&
$model/conv2d_6/Conv2D/ReadVariableOp?
model/conv2d_6/Conv2DConv2D#model/concatenate_5/concat:output:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model/conv2d_6/Conv2D?
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_6/BiasAdd/ReadVariableOp?
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/conv2d_6/BiasAdd?
$model/conv2d_6/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/conv2d_6/Max/reduction_indices?
model/conv2d_6/MaxMaxmodel/conv2d_6/BiasAdd:output:0-model/conv2d_6/Max/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
model/conv2d_6/Max?
model/conv2d_6/subSubmodel/conv2d_6/BiasAdd:output:0model/conv2d_6/Max:output:0*
T0*1
_output_shapes
:???????????2
model/conv2d_6/sub?
model/conv2d_6/ExpExpmodel/conv2d_6/sub:z:0*
T0*1
_output_shapes
:???????????2
model/conv2d_6/Exp?
$model/conv2d_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/conv2d_6/Sum/reduction_indices?
model/conv2d_6/SumSummodel/conv2d_6/Exp:y:0-model/conv2d_6/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
model/conv2d_6/Sum?
model/conv2d_6/truedivRealDivmodel/conv2d_6/Exp:y:0model/conv2d_6/Sum:output:0*
T0*1
_output_shapes
:???????????2
model/conv2d_6/truediv?!
IdentityIdentitymodel/conv2d_6/truediv:z:0&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp)^model/conv2d_fixed/Conv2D/ReadVariableOpG^model/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOpI^model/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_16^model/conv2d_fixed/batch_normalization/ReadVariableOp8^model/conv2d_fixed/batch_normalization/ReadVariableOp_1+^model/conv2d_fixed_1/Conv2D/ReadVariableOpK^model/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpM^model/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:^model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp<^model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1+^model/conv2d_fixed_2/Conv2D/ReadVariableOpK^model/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpM^model/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:^model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp<^model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1T^model/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpV^model/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1C^model/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOpE^model/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1>^model/conv2d_fixed__transpose/conv2d_transpose/ReadVariableOpW^model/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpY^model/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1F^model/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOpH^model/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1@^model/conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp#^model/conv2d_na/add/ReadVariableOpF^model/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOpH^model/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_15^model/conv2d_na/batch_normalization_1/ReadVariableOp7^model/conv2d_na/batch_normalization_1/ReadVariableOp_1.^model/conv2d_na/conv2d/BiasAdd/ReadVariableOp-^model/conv2d_na/conv2d/Conv2D/ReadVariableOp%^model/conv2d_na_1/add/ReadVariableOpH^model/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpJ^model/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17^model/conv2d_na_1/batch_normalization_3/ReadVariableOp9^model/conv2d_na_1/batch_normalization_3/ReadVariableOp_12^model/conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp1^model/conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp%^model/conv2d_na_2/add/ReadVariableOpH^model/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpJ^model/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17^model/conv2d_na_2/batch_normalization_5/ReadVariableOp9^model/conv2d_na_2/batch_normalization_5/ReadVariableOp_12^model/conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp1^model/conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp%^model/conv2d_na_3/add/ReadVariableOpH^model/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpJ^model/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17^model/conv2d_na_3/batch_normalization_6/ReadVariableOp9^model/conv2d_na_3/batch_normalization_6/ReadVariableOp_12^model/conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp1^model/conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp%^model/conv2d_na_4/add/ReadVariableOpH^model/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpJ^model/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17^model/conv2d_na_4/batch_normalization_7/ReadVariableOp9^model/conv2d_na_4/batch_normalization_7/ReadVariableOp_12^model/conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp1^model/conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp%^model/conv2d_na_5/add/ReadVariableOpH^model/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpJ^model/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17^model/conv2d_na_5/batch_normalization_9/ReadVariableOp9^model/conv2d_na_5/batch_normalization_9/ReadVariableOp_12^model/conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp1^model/conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2T
(model/conv2d_fixed/Conv2D/ReadVariableOp(model/conv2d_fixed/Conv2D/ReadVariableOp2?
Fmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOpFmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Hmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Hmodel/conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_12n
5model/conv2d_fixed/batch_normalization/ReadVariableOp5model/conv2d_fixed/batch_normalization/ReadVariableOp2r
7model/conv2d_fixed/batch_normalization/ReadVariableOp_17model/conv2d_fixed/batch_normalization/ReadVariableOp_12X
*model/conv2d_fixed_1/Conv2D/ReadVariableOp*model/conv2d_fixed_1/Conv2D/ReadVariableOp2?
Jmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpJmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Lmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Lmodel/conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12v
9model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp9model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp2z
;model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1;model/conv2d_fixed_1/batch_normalization_2/ReadVariableOp_12X
*model/conv2d_fixed_2/Conv2D/ReadVariableOp*model/conv2d_fixed_2/Conv2D/ReadVariableOp2?
Jmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpJmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Lmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Lmodel/conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12v
9model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp9model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp2z
;model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1;model/conv2d_fixed_2/batch_normalization_4/ReadVariableOp_12?
Smodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpSmodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Umodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Umodel/conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12?
Bmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOpBmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp2?
Dmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1Dmodel/conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_12~
=model/conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp=model/conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp2?
Vmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpVmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Xmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Xmodel/conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12?
Emodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOpEmodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp2?
Gmodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1Gmodel/conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_12?
?model/conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp?model/conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp2H
"model/conv2d_na/add/ReadVariableOp"model/conv2d_na/add/ReadVariableOp2?
Emodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOpEmodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Gmodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Gmodel/conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12l
4model/conv2d_na/batch_normalization_1/ReadVariableOp4model/conv2d_na/batch_normalization_1/ReadVariableOp2p
6model/conv2d_na/batch_normalization_1/ReadVariableOp_16model/conv2d_na/batch_normalization_1/ReadVariableOp_12^
-model/conv2d_na/conv2d/BiasAdd/ReadVariableOp-model/conv2d_na/conv2d/BiasAdd/ReadVariableOp2\
,model/conv2d_na/conv2d/Conv2D/ReadVariableOp,model/conv2d_na/conv2d/Conv2D/ReadVariableOp2L
$model/conv2d_na_1/add/ReadVariableOp$model/conv2d_na_1/add/ReadVariableOp2?
Gmodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpGmodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Imodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Imodel/conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12p
6model/conv2d_na_1/batch_normalization_3/ReadVariableOp6model/conv2d_na_1/batch_normalization_3/ReadVariableOp2t
8model/conv2d_na_1/batch_normalization_3/ReadVariableOp_18model/conv2d_na_1/batch_normalization_3/ReadVariableOp_12f
1model/conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp1model/conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp2d
0model/conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp0model/conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp2L
$model/conv2d_na_2/add/ReadVariableOp$model/conv2d_na_2/add/ReadVariableOp2?
Gmodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpGmodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Imodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Imodel/conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12p
6model/conv2d_na_2/batch_normalization_5/ReadVariableOp6model/conv2d_na_2/batch_normalization_5/ReadVariableOp2t
8model/conv2d_na_2/batch_normalization_5/ReadVariableOp_18model/conv2d_na_2/batch_normalization_5/ReadVariableOp_12f
1model/conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp1model/conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp2d
0model/conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp0model/conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp2L
$model/conv2d_na_3/add/ReadVariableOp$model/conv2d_na_3/add/ReadVariableOp2?
Gmodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpGmodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Imodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Imodel/conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12p
6model/conv2d_na_3/batch_normalization_6/ReadVariableOp6model/conv2d_na_3/batch_normalization_6/ReadVariableOp2t
8model/conv2d_na_3/batch_normalization_6/ReadVariableOp_18model/conv2d_na_3/batch_normalization_6/ReadVariableOp_12f
1model/conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp1model/conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp2d
0model/conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp0model/conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp2L
$model/conv2d_na_4/add/ReadVariableOp$model/conv2d_na_4/add/ReadVariableOp2?
Gmodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpGmodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Imodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Imodel/conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12p
6model/conv2d_na_4/batch_normalization_7/ReadVariableOp6model/conv2d_na_4/batch_normalization_7/ReadVariableOp2t
8model/conv2d_na_4/batch_normalization_7/ReadVariableOp_18model/conv2d_na_4/batch_normalization_7/ReadVariableOp_12f
1model/conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp1model/conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp2d
0model/conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp0model/conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp2L
$model/conv2d_na_5/add/ReadVariableOp$model/conv2d_na_5/add/ReadVariableOp2?
Gmodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpGmodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Imodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Imodel/conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12p
6model/conv2d_na_5/batch_normalization_9/ReadVariableOp6model/conv2d_na_5/batch_normalization_9/ReadVariableOp2t
8model/conv2d_na_5/batch_normalization_9/ReadVariableOp_18model/conv2d_na_5/batch_normalization_9/ReadVariableOp_12f
1model/conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp1model/conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp2d
0model/conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp0model/conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
__inference_loss_fn_5_7155J
Fconv2d_na_4_conv2d_4_kernel_regularizer_square_readvariableop_resource
identity??=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconv2d_na_4_conv2d_4_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
IdentityIdentity/conv2d_na_4/conv2d_4/kernel/Regularizer/mul:z:0>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7320

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_2_layer_call_fn_5994

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_27972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P&::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P&
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_3014

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_4/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_7/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P`::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P`
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_6848J
Fconv2d_na_1_conv2d_1_kernel_regularizer_square_readvariableop_resource
identity??=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconv2d_na_1_conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
IdentityIdentity/conv2d_na_1/conv2d_1/kernel/Regularizer/mul:z:0>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp
?
?
4__inference_batch_normalization_5_layer_call_fn_6969

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
-__inference_conv2d_fixed_1_layer_call_fn_5663

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_24462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_fixed_2_layer_call_fn_5873

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_26682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
X
,__inference_concatenate_1_layer_call_fn_5691
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_24982
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????Z?:?????????Z?:Z V
0
_output_shapes
:?????????Z?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????Z?
"
_user_specified_name
inputs/1
?
?
$__inference_model_layer_call_fn_4350
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_42212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6956

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_2542

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_1/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6578

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_6156

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_4/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_7/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P`::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P`
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_5732

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_1/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs
?
?
6__inference_conv2d_fixed__transpose_layer_call_fn_6299

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_31422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????-P :::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7100

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6938

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1512

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_8_layer_call_fn_7219

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_19592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?D
?__inference_model_layer_call_and_return_conditional_losses_5147

inputs/
+conv2d_fixed_conv2d_readvariableop_resource<
8conv2d_fixed_batch_normalization_readvariableop_resource>
:conv2d_fixed_batch_normalization_readvariableop_1_resourceM
Iconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_resourceO
Kconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/conv2d_na_conv2d_conv2d_readvariableop_resource4
0conv2d_na_conv2d_biasadd_readvariableop_resource;
7conv2d_na_batch_normalization_1_readvariableop_resource=
9conv2d_na_batch_normalization_1_readvariableop_1_resourceL
Hconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceN
Jconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource1
-conv2d_fixed_1_conv2d_readvariableop_resource@
<conv2d_fixed_1_batch_normalization_2_readvariableop_resourceB
>conv2d_fixed_1_batch_normalization_2_readvariableop_1_resourceQ
Mconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceS
Oconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_1_conv2d_1_conv2d_readvariableop_resource8
4conv2d_na_1_conv2d_1_biasadd_readvariableop_resource=
9conv2d_na_1_batch_normalization_3_readvariableop_resource?
;conv2d_na_1_batch_normalization_3_readvariableop_1_resourceN
Jconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource1
-conv2d_fixed_2_conv2d_readvariableop_resource@
<conv2d_fixed_2_batch_normalization_4_readvariableop_resourceB
>conv2d_fixed_2_batch_normalization_4_readvariableop_1_resourceQ
Mconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceS
Oconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_2_conv2d_2_conv2d_readvariableop_resource8
4conv2d_na_2_conv2d_2_biasadd_readvariableop_resource=
9conv2d_na_2_batch_normalization_5_readvariableop_resource?
;conv2d_na_2_batch_normalization_5_readvariableop_1_resourceN
Jconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_3_conv2d_3_conv2d_readvariableop_resource8
4conv2d_na_3_conv2d_3_biasadd_readvariableop_resource=
9conv2d_na_3_batch_normalization_6_readvariableop_resource?
;conv2d_na_3_batch_normalization_6_readvariableop_1_resourceN
Jconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_4_conv2d_4_conv2d_readvariableop_resource8
4conv2d_na_4_conv2d_4_biasadd_readvariableop_resource=
9conv2d_na_4_batch_normalization_7_readvariableop_resource?
;conv2d_na_4_batch_normalization_7_readvariableop_1_resourceN
Jconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceD
@conv2d_fixed__transpose_conv2d_transpose_readvariableop_resourceI
Econv2d_fixed__transpose_batch_normalization_8_readvariableop_resourceK
Gconv2d_fixed__transpose_batch_normalization_8_readvariableop_1_resourceZ
Vconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_resource\
Xconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_5_conv2d_5_conv2d_readvariableop_resource8
4conv2d_na_5_conv2d_5_biasadd_readvariableop_resource=
9conv2d_na_5_batch_normalization_9_readvariableop_resource?
;conv2d_na_5_batch_normalization_9_readvariableop_1_resourceN
Jconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceF
Bconv2d_fixed__transpose_1_conv2d_transpose_readvariableop_resourceL
Hconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_resourceN
Jconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_1_resource]
Yconv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource_
[conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?"conv2d_fixed/Conv2D/ReadVariableOp?@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp?Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?/conv2d_fixed/batch_normalization/ReadVariableOp?1conv2d_fixed/batch_normalization/ReadVariableOp_1?$conv2d_fixed_1/Conv2D/ReadVariableOp?Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?3conv2d_fixed_1/batch_normalization_2/ReadVariableOp?5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1?$conv2d_fixed_2/Conv2D/ReadVariableOp?Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?3conv2d_fixed_2/batch_normalization_4/ReadVariableOp?5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1?Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1?7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp?Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1??conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1?9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp?conv2d_na/add/ReadVariableOp??conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?.conv2d_na/batch_normalization_1/ReadVariableOp?0conv2d_na/batch_normalization_1/ReadVariableOp_1?'conv2d_na/conv2d/BiasAdd/ReadVariableOp?&conv2d_na/conv2d/Conv2D/ReadVariableOp?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_1/add/ReadVariableOp?Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_1/batch_normalization_3/ReadVariableOp?2conv2d_na_1/batch_normalization_3/ReadVariableOp_1?+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp?*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_2/add/ReadVariableOp?Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_2/batch_normalization_5/ReadVariableOp?2conv2d_na_2/batch_normalization_5/ReadVariableOp_1?+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp?*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_3/add/ReadVariableOp?Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_3/batch_normalization_6/ReadVariableOp?2conv2d_na_3/batch_normalization_6/ReadVariableOp_1?+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp?*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_4/add/ReadVariableOp?Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_4/batch_normalization_7/ReadVariableOp?2conv2d_na_4/batch_normalization_7/ReadVariableOp_1?+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp?*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_5/add/ReadVariableOp?Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_5/batch_normalization_9/ReadVariableOp?2conv2d_na_5/batch_normalization_9/ReadVariableOp_1?+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp?*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_fixed/Conv2D/ReadVariableOpReadVariableOp+conv2d_fixed_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"conv2d_fixed/Conv2D/ReadVariableOp?
conv2d_fixed/Conv2DConv2Dinputs*conv2d_fixed/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_fixed/Conv2D?
/conv2d_fixed/batch_normalization/ReadVariableOpReadVariableOp8conv2d_fixed_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_fixed/batch_normalization/ReadVariableOp?
1conv2d_fixed/batch_normalization/ReadVariableOp_1ReadVariableOp:conv2d_fixed_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype023
1conv2d_fixed/batch_normalization/ReadVariableOp_1?
@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpIconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
1conv2d_fixed/batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_fixed/Conv2D:output:07conv2d_fixed/batch_normalization/ReadVariableOp:value:09conv2d_fixed/batch_normalization/ReadVariableOp_1:value:0Hconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 23
1conv2d_fixed/batch_normalization/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV25conv2d_fixed/batch_normalization/FusedBatchNormV3:y:0inputs concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
&conv2d_na/conv2d/Conv2D/ReadVariableOpReadVariableOp/conv2d_na_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&conv2d_na/conv2d/Conv2D/ReadVariableOp?
conv2d_na/conv2d/Conv2DConv2Dconcatenate/concat:output:0.conv2d_na/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_na/conv2d/Conv2D?
'conv2d_na/conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_na_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_na/conv2d/BiasAdd/ReadVariableOp?
conv2d_na/conv2d/BiasAddBiasAdd conv2d_na/conv2d/Conv2D:output:0/conv2d_na/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_na/conv2d/BiasAdd?
.conv2d_na/batch_normalization_1/ReadVariableOpReadVariableOp7conv2d_na_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype020
.conv2d_na/batch_normalization_1/ReadVariableOp?
0conv2d_na/batch_normalization_1/ReadVariableOp_1ReadVariableOp9conv2d_na_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype022
0conv2d_na/batch_normalization_1/ReadVariableOp_1?
?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
0conv2d_na/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!conv2d_na/conv2d/BiasAdd:output:06conv2d_na/batch_normalization_1/ReadVariableOp:value:08conv2d_na/batch_normalization_1/ReadVariableOp_1:value:0Gconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Iconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 22
0conv2d_na/batch_normalization_1/FusedBatchNormV3?
conv2d_na/add/ReadVariableOpReadVariableOp0conv2d_na_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d_na/add/ReadVariableOp?
conv2d_na/addAddV24conv2d_na/batch_normalization_1/FusedBatchNormV3:y:0$conv2d_na/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_na/addw
conv2d_na/ReluReluconv2d_na/add:z:0*
T0*1
_output_shapes
:???????????2
conv2d_na/Relu?
$conv2d_fixed_1/Conv2D/ReadVariableOpReadVariableOp-conv2d_fixed_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$conv2d_fixed_1/Conv2D/ReadVariableOp?
conv2d_fixed_1/Conv2DConv2Dconv2d_na/Relu:activations:0,conv2d_fixed_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_fixed_1/Conv2D?
3conv2d_fixed_1/batch_normalization_2/ReadVariableOpReadVariableOp<conv2d_fixed_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype025
3conv2d_fixed_1/batch_normalization_2/ReadVariableOp?
5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp>conv2d_fixed_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype027
5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1?
Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpMconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
5conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_fixed_1/Conv2D:output:0;conv2d_fixed_1/batch_normalization_2/ReadVariableOp:value:0=conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1:value:0Lconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Nconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z?:::::*
epsilon%o?:*
is_training( 27
5conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3?
tf.image.resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Z   ?   2
tf.image.resize/resize/size?
%tf.image.resize/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0$tf.image.resize/resize/size:output:0*
T0*0
_output_shapes
:?????????Z?*
half_pixel_centers(2'
%tf.image.resize/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV29conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3:y:06tf.image.resize/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?2
concatenate_1/concat?
*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp?
conv2d_na_1/conv2d_1/Conv2DConv2Dconcatenate_1/concat:output:02conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_na_1/conv2d_1/Conv2D?
+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp?
conv2d_na_1/conv2d_1/BiasAddBiasAdd$conv2d_na_1/conv2d_1/Conv2D:output:03conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_1/conv2d_1/BiasAdd?
0conv2d_na_1/batch_normalization_3/ReadVariableOpReadVariableOp9conv2d_na_1_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_1/batch_normalization_3/ReadVariableOp?
2conv2d_na_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp;conv2d_na_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_1/batch_normalization_3/ReadVariableOp_1?
Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%conv2d_na_1/conv2d_1/BiasAdd:output:08conv2d_na_1/batch_normalization_3/ReadVariableOp:value:0:conv2d_na_1/batch_normalization_3/ReadVariableOp_1:value:0Iconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 24
2conv2d_na_1/batch_normalization_3/FusedBatchNormV3?
conv2d_na_1/add/ReadVariableOpReadVariableOp4conv2d_na_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_1/add/ReadVariableOp?
conv2d_na_1/addAddV26conv2d_na_1/batch_normalization_3/FusedBatchNormV3:y:0&conv2d_na_1/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_1/add|
conv2d_na_1/ReluReluconv2d_na_1/add:z:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_1/Relu?
$conv2d_fixed_2/Conv2D/ReadVariableOpReadVariableOp-conv2d_fixed_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$conv2d_fixed_2/Conv2D/ReadVariableOp?
conv2d_fixed_2/Conv2DConv2Dconv2d_na_1/Relu:activations:0,conv2d_fixed_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_fixed_2/Conv2D?
3conv2d_fixed_2/batch_normalization_4/ReadVariableOpReadVariableOp<conv2d_fixed_2_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype025
3conv2d_fixed_2/batch_normalization_4/ReadVariableOp?
5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1ReadVariableOp>conv2d_fixed_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype027
5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1?
Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpMconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
5conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_fixed_2/Conv2D:output:0;conv2d_fixed_2/batch_normalization_4/ReadVariableOp:value:0=conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1:value:0Lconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Nconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 27
5conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3?
tf.image.resize_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"-   P   2
tf.image.resize_1/resize/size?
'tf.image.resize_1/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0&tf.image.resize_1/resize/size:output:0*
T0*/
_output_shapes
:?????????-P*
half_pixel_centers(2)
'tf.image.resize_1/resize/ResizeBilinearx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV29conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3:y:08tf.image.resize_1/resize/ResizeBilinear:resized_images:0"concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P&2
concatenate_2/concat?
*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02,
*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp?
conv2d_na_2/conv2d_2/Conv2DConv2Dconcatenate_2/concat:output:02conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_na_2/conv2d_2/Conv2D?
+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp?
conv2d_na_2/conv2d_2/BiasAddBiasAdd$conv2d_na_2/conv2d_2/Conv2D:output:03conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_2/conv2d_2/BiasAdd?
0conv2d_na_2/batch_normalization_5/ReadVariableOpReadVariableOp9conv2d_na_2_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_2/batch_normalization_5/ReadVariableOp?
2conv2d_na_2/batch_normalization_5/ReadVariableOp_1ReadVariableOp;conv2d_na_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_2/batch_normalization_5/ReadVariableOp_1?
Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%conv2d_na_2/conv2d_2/BiasAdd:output:08conv2d_na_2/batch_normalization_5/ReadVariableOp:value:0:conv2d_na_2/batch_normalization_5/ReadVariableOp_1:value:0Iconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 24
2conv2d_na_2/batch_normalization_5/FusedBatchNormV3?
conv2d_na_2/add/ReadVariableOpReadVariableOp4conv2d_na_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_2/add/ReadVariableOp?
conv2d_na_2/addAddV26conv2d_na_2/batch_normalization_5/FusedBatchNormV3:y:0&conv2d_na_2/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_2/add{
conv2d_na_2/ReluReluconv2d_na_2/add:z:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_2/Relu?
*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp?
conv2d_na_3/conv2d_3/Conv2DConv2Dconv2d_na_2/Relu:activations:02conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@*
paddingSAME*
strides
2
conv2d_na_3/conv2d_3/Conv2D?
+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp?
conv2d_na_3/conv2d_3/BiasAddBiasAdd$conv2d_na_3/conv2d_3/Conv2D:output:03conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_na_3/conv2d_3/BiasAdd?
0conv2d_na_3/batch_normalization_6/ReadVariableOpReadVariableOp9conv2d_na_3_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_na_3/batch_normalization_6/ReadVariableOp?
2conv2d_na_3/batch_normalization_6/ReadVariableOp_1ReadVariableOp;conv2d_na_3_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2conv2d_na_3/batch_normalization_6/ReadVariableOp_1?
Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_3/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%conv2d_na_3/conv2d_3/BiasAdd:output:08conv2d_na_3/batch_normalization_6/ReadVariableOp:value:0:conv2d_na_3/batch_normalization_6/ReadVariableOp_1:value:0Iconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2conv2d_na_3/batch_normalization_6/FusedBatchNormV3?
conv2d_na_3/add/ReadVariableOpReadVariableOp4conv2d_na_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv2d_na_3/add/ReadVariableOp?
conv2d_na_3/addAddV26conv2d_na_3/batch_normalization_6/FusedBatchNormV3:y:0&conv2d_na_3/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_na_3/add{
conv2d_na_3/ReluReluconv2d_na_3/add:z:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_na_3/Relux
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2conv2d_na_2/Relu:activations:0conv2d_na_3/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P`2
concatenate_3/concat?
*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02,
*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp?
conv2d_na_4/conv2d_4/Conv2DConv2Dconcatenate_3/concat:output:02conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_na_4/conv2d_4/Conv2D?
+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp?
conv2d_na_4/conv2d_4/BiasAddBiasAdd$conv2d_na_4/conv2d_4/Conv2D:output:03conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_4/conv2d_4/BiasAdd?
0conv2d_na_4/batch_normalization_7/ReadVariableOpReadVariableOp9conv2d_na_4_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_4/batch_normalization_7/ReadVariableOp?
2conv2d_na_4/batch_normalization_7/ReadVariableOp_1ReadVariableOp;conv2d_na_4_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_4/batch_normalization_7/ReadVariableOp_1?
Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_4/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%conv2d_na_4/conv2d_4/BiasAdd:output:08conv2d_na_4/batch_normalization_7/ReadVariableOp:value:0:conv2d_na_4/batch_normalization_7/ReadVariableOp_1:value:0Iconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 24
2conv2d_na_4/batch_normalization_7/FusedBatchNormV3?
conv2d_na_4/add/ReadVariableOpReadVariableOp4conv2d_na_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_4/add/ReadVariableOp?
conv2d_na_4/addAddV26conv2d_na_4/batch_normalization_7/FusedBatchNormV3:y:0&conv2d_na_4/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_4/add{
conv2d_na_4/ReluReluconv2d_na_4/add:z:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_4/Relu?
4conv2d_fixed__transpose/conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   Z   ?       26
4conv2d_fixed__transpose/conv2d_transpose/input_sizes?
7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOpReadVariableOp@conv2d_fixed__transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype029
7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp?
(conv2d_fixed__transpose/conv2d_transposeConv2DBackpropInput=conv2d_fixed__transpose/conv2d_transpose/input_sizes:output:0?conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_na_4/Relu:activations:0*
T0*0
_output_shapes
:?????????Z? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2*
(conv2d_fixed__transpose/conv2d_transpose?
<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOpReadVariableOpEconv2d_fixed__transpose_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02>
<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?
>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1ReadVariableOpGconv2d_fixed__transpose_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1?
Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpVconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02O
Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
>conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3FusedBatchNormV31conv2d_fixed__transpose/conv2d_transpose:output:0Dconv2d_fixed__transpose/batch_normalization_8/ReadVariableOp:value:0Fconv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1:value:0Uconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Wconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2@
>conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3x
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2conv2d_na_1/Relu:activations:0Bconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3:y:0"concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?@2
concatenate_4/concat?
*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02,
*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp?
conv2d_na_5/conv2d_5/Conv2DConv2Dconcatenate_4/concat:output:02conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_na_5/conv2d_5/Conv2D?
+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp?
conv2d_na_5/conv2d_5/BiasAddBiasAdd$conv2d_na_5/conv2d_5/Conv2D:output:03conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_5/conv2d_5/BiasAdd?
0conv2d_na_5/batch_normalization_9/ReadVariableOpReadVariableOp9conv2d_na_5_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_5/batch_normalization_9/ReadVariableOp?
2conv2d_na_5/batch_normalization_9/ReadVariableOp_1ReadVariableOp;conv2d_na_5_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_5/batch_normalization_9/ReadVariableOp_1?
Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_5/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%conv2d_na_5/conv2d_5/BiasAdd:output:08conv2d_na_5/batch_normalization_9/ReadVariableOp:value:0:conv2d_na_5/batch_normalization_9/ReadVariableOp_1:value:0Iconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 24
2conv2d_na_5/batch_normalization_9/FusedBatchNormV3?
conv2d_na_5/add/ReadVariableOpReadVariableOp4conv2d_na_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_5/add/ReadVariableOp?
conv2d_na_5/addAddV26conv2d_na_5/batch_normalization_9/FusedBatchNormV3:y:0&conv2d_na_5/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_5/add|
conv2d_na_5/ReluReluconv2d_na_5/add:z:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_5/Relu?
6conv2d_fixed__transpose_1/conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   ?   @      28
6conv2d_fixed__transpose_1/conv2d_transpose/input_sizes?
9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpBconv2d_fixed__transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02;
9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp?
*conv2d_fixed__transpose_1/conv2d_transposeConv2DBackpropInput?conv2d_fixed__transpose_1/conv2d_transpose/input_sizes:output:0Aconv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp:value:0conv2d_na_5/Relu:activations:0*
T0*1
_output_shapes
:??????????? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2,
*conv2d_fixed__transpose_1/conv2d_transpose?
?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOpReadVariableOpHconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02A
?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?
Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1ReadVariableOpJconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02C
Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1?
Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYconv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02R
Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02T
Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
Aconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV33conv2d_fixed__transpose_1/conv2d_transpose:output:0Gconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp:value:0Iconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1:value:0Xconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Zconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2C
Aconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3x
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2conv2d_na/Relu:activations:0Econv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3:y:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????(2
concatenate_5/concat?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dconcatenate_5/concat:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_6/BiasAdd?
conv2d_6/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv2d_6/Max/reduction_indices?
conv2d_6/MaxMaxconv2d_6/BiasAdd:output:0'conv2d_6/Max/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
conv2d_6/Max?
conv2d_6/subSubconv2d_6/BiasAdd:output:0conv2d_6/Max:output:0*
T0*1
_output_shapes
:???????????2
conv2d_6/subq
conv2d_6/ExpExpconv2d_6/sub:z:0*
T0*1
_output_shapes
:???????????2
conv2d_6/Exp?
conv2d_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv2d_6/Sum/reduction_indices?
conv2d_6/SumSumconv2d_6/Exp:y:0'conv2d_6/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
conv2d_6/Sum?
conv2d_6/truedivRealDivconv2d_6/Exp:y:0conv2d_6/Sum:output:0*
T0*1
_output_shapes
:???????????2
conv2d_6/truediv?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/conv2d_na_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?"
IdentityIdentityconv2d_6/truediv:z:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp#^conv2d_fixed/Conv2D/ReadVariableOpA^conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOpC^conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_10^conv2d_fixed/batch_normalization/ReadVariableOp2^conv2d_fixed/batch_normalization/ReadVariableOp_1%^conv2d_fixed_1/Conv2D/ReadVariableOpE^conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpG^conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_14^conv2d_fixed_1/batch_normalization_2/ReadVariableOp6^conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1%^conv2d_fixed_2/Conv2D/ReadVariableOpE^conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpG^conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_14^conv2d_fixed_2/batch_normalization_4/ReadVariableOp6^conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1N^conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpP^conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=^conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?^conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_18^conv2d_fixed__transpose/conv2d_transpose/ReadVariableOpQ^conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpS^conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@^conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOpB^conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1:^conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp^conv2d_na/add/ReadVariableOp@^conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOpB^conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/^conv2d_na/batch_normalization_1/ReadVariableOp1^conv2d_na/batch_normalization_1/ReadVariableOp_1(^conv2d_na/conv2d/BiasAdd/ReadVariableOp'^conv2d_na/conv2d/Conv2D/ReadVariableOp:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_1/add/ReadVariableOpB^conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_1/batch_normalization_3/ReadVariableOp3^conv2d_na_1/batch_normalization_3/ReadVariableOp_1,^conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp+^conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_2/add/ReadVariableOpB^conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_2/batch_normalization_5/ReadVariableOp3^conv2d_na_2/batch_normalization_5/ReadVariableOp_1,^conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp+^conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_3/add/ReadVariableOpB^conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_3/batch_normalization_6/ReadVariableOp3^conv2d_na_3/batch_normalization_6/ReadVariableOp_1,^conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp+^conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_4/add/ReadVariableOpB^conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_4/batch_normalization_7/ReadVariableOp3^conv2d_na_4/batch_normalization_7/ReadVariableOp_1,^conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp+^conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_5/add/ReadVariableOpB^conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpD^conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_5/batch_normalization_9/ReadVariableOp3^conv2d_na_5/batch_normalization_9/ReadVariableOp_1,^conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp+^conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2H
"conv2d_fixed/Conv2D/ReadVariableOp"conv2d_fixed/Conv2D/ReadVariableOp2?
@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_12b
/conv2d_fixed/batch_normalization/ReadVariableOp/conv2d_fixed/batch_normalization/ReadVariableOp2f
1conv2d_fixed/batch_normalization/ReadVariableOp_11conv2d_fixed/batch_normalization/ReadVariableOp_12L
$conv2d_fixed_1/Conv2D/ReadVariableOp$conv2d_fixed_1/Conv2D/ReadVariableOp2?
Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpDconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12j
3conv2d_fixed_1/batch_normalization_2/ReadVariableOp3conv2d_fixed_1/batch_normalization_2/ReadVariableOp2n
5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_15conv2d_fixed_1/batch_normalization_2/ReadVariableOp_12L
$conv2d_fixed_2/Conv2D/ReadVariableOp$conv2d_fixed_2/Conv2D/ReadVariableOp2?
Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpDconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12j
3conv2d_fixed_2/batch_normalization_4/ReadVariableOp3conv2d_fixed_2/batch_normalization_4/ReadVariableOp2n
5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_15conv2d_fixed_2/batch_normalization_4/ReadVariableOp_12?
Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpMconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12|
<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp2?
>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_12r
7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp2?
Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpPconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12?
?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp2?
Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_12v
9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp2<
conv2d_na/add/ReadVariableOpconv2d_na/add/ReadVariableOp2?
?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12`
.conv2d_na/batch_normalization_1/ReadVariableOp.conv2d_na/batch_normalization_1/ReadVariableOp2d
0conv2d_na/batch_normalization_1/ReadVariableOp_10conv2d_na/batch_normalization_1/ReadVariableOp_12R
'conv2d_na/conv2d/BiasAdd/ReadVariableOp'conv2d_na/conv2d/BiasAdd/ReadVariableOp2P
&conv2d_na/conv2d/Conv2D/ReadVariableOp&conv2d_na/conv2d/Conv2D/ReadVariableOp2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_1/add/ReadVariableOpconv2d_na_1/add/ReadVariableOp2?
Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_1/batch_normalization_3/ReadVariableOp0conv2d_na_1/batch_normalization_3/ReadVariableOp2h
2conv2d_na_1/batch_normalization_3/ReadVariableOp_12conv2d_na_1/batch_normalization_3/ReadVariableOp_12Z
+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp2X
*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_2/add/ReadVariableOpconv2d_na_2/add/ReadVariableOp2?
Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_2/batch_normalization_5/ReadVariableOp0conv2d_na_2/batch_normalization_5/ReadVariableOp2h
2conv2d_na_2/batch_normalization_5/ReadVariableOp_12conv2d_na_2/batch_normalization_5/ReadVariableOp_12Z
+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp2X
*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_3/add/ReadVariableOpconv2d_na_3/add/ReadVariableOp2?
Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_3/batch_normalization_6/ReadVariableOp0conv2d_na_3/batch_normalization_6/ReadVariableOp2h
2conv2d_na_3/batch_normalization_6/ReadVariableOp_12conv2d_na_3/batch_normalization_6/ReadVariableOp_12Z
+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp2X
*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_4/add/ReadVariableOpconv2d_na_4/add/ReadVariableOp2?
Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_4/batch_normalization_7/ReadVariableOp0conv2d_na_4/batch_normalization_7/ReadVariableOp2h
2conv2d_na_4/batch_normalization_7/ReadVariableOp_12conv2d_na_4/batch_normalization_7/ReadVariableOp_12Z
+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp2X
*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_5/add/ReadVariableOpconv2d_na_5/add/ReadVariableOp2?
Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpAconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_5/batch_normalization_9/ReadVariableOp0conv2d_na_5/batch_normalization_9/ReadVariableOp2h
2conv2d_na_5/batch_normalization_9/ReadVariableOp_12conv2d_na_5/batch_normalization_9/ReadVariableOp_12Z
+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp2X
*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7193

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
X
,__inference_concatenate_4_layer_call_fn_6312
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_31922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????Z?@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????Z? :?????????Z? :Z V
0
_output_shapes
:?????????Z? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????Z? 
"
_user_specified_name
inputs/1
?-
?
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_2575

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_1/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_4_layer_call_fn_6912

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_15432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_6068

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_3/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P@2
Relu?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P ::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6596

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?#
?
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_5822

inputs"
conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_1:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1200

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1408

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?6
?
C__inference_conv2d_na_layer_call_and_return_conditional_losses_5537

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
add/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
addY
ReluReluadd:z:0*
T0*1
_output_shapes
:???????????2
Relu?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:???????????::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_3_layer_call_fn_6824

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_2_layer_call_fn_6767

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?#
?
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_5627

inputs"
conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z?:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_3269

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_5/BiasAdd?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_9/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?@::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?@
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_5765

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_1/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1824

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?M
?__inference_model_layer_call_and_return_conditional_losses_4847

inputs/
+conv2d_fixed_conv2d_readvariableop_resource<
8conv2d_fixed_batch_normalization_readvariableop_resource>
:conv2d_fixed_batch_normalization_readvariableop_1_resourceM
Iconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_resourceO
Kconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/conv2d_na_conv2d_conv2d_readvariableop_resource4
0conv2d_na_conv2d_biasadd_readvariableop_resource;
7conv2d_na_batch_normalization_1_readvariableop_resource=
9conv2d_na_batch_normalization_1_readvariableop_1_resourceL
Hconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceN
Jconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource1
-conv2d_fixed_1_conv2d_readvariableop_resource@
<conv2d_fixed_1_batch_normalization_2_readvariableop_resourceB
>conv2d_fixed_1_batch_normalization_2_readvariableop_1_resourceQ
Mconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceS
Oconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_1_conv2d_1_conv2d_readvariableop_resource8
4conv2d_na_1_conv2d_1_biasadd_readvariableop_resource=
9conv2d_na_1_batch_normalization_3_readvariableop_resource?
;conv2d_na_1_batch_normalization_3_readvariableop_1_resourceN
Jconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource1
-conv2d_fixed_2_conv2d_readvariableop_resource@
<conv2d_fixed_2_batch_normalization_4_readvariableop_resourceB
>conv2d_fixed_2_batch_normalization_4_readvariableop_1_resourceQ
Mconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceS
Oconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_2_conv2d_2_conv2d_readvariableop_resource8
4conv2d_na_2_conv2d_2_biasadd_readvariableop_resource=
9conv2d_na_2_batch_normalization_5_readvariableop_resource?
;conv2d_na_2_batch_normalization_5_readvariableop_1_resourceN
Jconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_3_conv2d_3_conv2d_readvariableop_resource8
4conv2d_na_3_conv2d_3_biasadd_readvariableop_resource=
9conv2d_na_3_batch_normalization_6_readvariableop_resource?
;conv2d_na_3_batch_normalization_6_readvariableop_1_resourceN
Jconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_4_conv2d_4_conv2d_readvariableop_resource8
4conv2d_na_4_conv2d_4_biasadd_readvariableop_resource=
9conv2d_na_4_batch_normalization_7_readvariableop_resource?
;conv2d_na_4_batch_normalization_7_readvariableop_1_resourceN
Jconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceD
@conv2d_fixed__transpose_conv2d_transpose_readvariableop_resourceI
Econv2d_fixed__transpose_batch_normalization_8_readvariableop_resourceK
Gconv2d_fixed__transpose_batch_normalization_8_readvariableop_1_resourceZ
Vconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_resource\
Xconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7
3conv2d_na_5_conv2d_5_conv2d_readvariableop_resource8
4conv2d_na_5_conv2d_5_biasadd_readvariableop_resource=
9conv2d_na_5_batch_normalization_9_readvariableop_resource?
;conv2d_na_5_batch_normalization_9_readvariableop_1_resourceN
Jconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceP
Lconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceF
Bconv2d_fixed__transpose_1_conv2d_transpose_readvariableop_resourceL
Hconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_resourceN
Jconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_1_resource]
Yconv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource_
[conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?"conv2d_fixed/Conv2D/ReadVariableOp?/conv2d_fixed/batch_normalization/AssignNewValue?1conv2d_fixed/batch_normalization/AssignNewValue_1?@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp?Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?/conv2d_fixed/batch_normalization/ReadVariableOp?1conv2d_fixed/batch_normalization/ReadVariableOp_1?$conv2d_fixed_1/Conv2D/ReadVariableOp?3conv2d_fixed_1/batch_normalization_2/AssignNewValue?5conv2d_fixed_1/batch_normalization_2/AssignNewValue_1?Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?3conv2d_fixed_1/batch_normalization_2/ReadVariableOp?5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1?$conv2d_fixed_2/Conv2D/ReadVariableOp?3conv2d_fixed_2/batch_normalization_4/AssignNewValue?5conv2d_fixed_2/batch_normalization_4/AssignNewValue_1?Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?3conv2d_fixed_2/batch_normalization_4/ReadVariableOp?5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1?<conv2d_fixed__transpose/batch_normalization_8/AssignNewValue?>conv2d_fixed__transpose/batch_normalization_8/AssignNewValue_1?Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1?7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp??conv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue?Aconv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue_1?Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1??conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1?9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp?conv2d_na/add/ReadVariableOp?.conv2d_na/batch_normalization_1/AssignNewValue?0conv2d_na/batch_normalization_1/AssignNewValue_1??conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?.conv2d_na/batch_normalization_1/ReadVariableOp?0conv2d_na/batch_normalization_1/ReadVariableOp_1?'conv2d_na/conv2d/BiasAdd/ReadVariableOp?&conv2d_na/conv2d/Conv2D/ReadVariableOp?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_1/add/ReadVariableOp?0conv2d_na_1/batch_normalization_3/AssignNewValue?2conv2d_na_1/batch_normalization_3/AssignNewValue_1?Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_1/batch_normalization_3/ReadVariableOp?2conv2d_na_1/batch_normalization_3/ReadVariableOp_1?+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp?*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_2/add/ReadVariableOp?0conv2d_na_2/batch_normalization_5/AssignNewValue?2conv2d_na_2/batch_normalization_5/AssignNewValue_1?Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_2/batch_normalization_5/ReadVariableOp?2conv2d_na_2/batch_normalization_5/ReadVariableOp_1?+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp?*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_3/add/ReadVariableOp?0conv2d_na_3/batch_normalization_6/AssignNewValue?2conv2d_na_3/batch_normalization_6/AssignNewValue_1?Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_3/batch_normalization_6/ReadVariableOp?2conv2d_na_3/batch_normalization_6/ReadVariableOp_1?+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp?*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_4/add/ReadVariableOp?0conv2d_na_4/batch_normalization_7/AssignNewValue?2conv2d_na_4/batch_normalization_7/AssignNewValue_1?Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_4/batch_normalization_7/ReadVariableOp?2conv2d_na_4/batch_normalization_7/ReadVariableOp_1?+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp?*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?conv2d_na_5/add/ReadVariableOp?0conv2d_na_5/batch_normalization_9/AssignNewValue?2conv2d_na_5/batch_normalization_9/AssignNewValue_1?Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?0conv2d_na_5/batch_normalization_9/ReadVariableOp?2conv2d_na_5/batch_normalization_9/ReadVariableOp_1?+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp?*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_fixed/Conv2D/ReadVariableOpReadVariableOp+conv2d_fixed_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"conv2d_fixed/Conv2D/ReadVariableOp?
conv2d_fixed/Conv2DConv2Dinputs*conv2d_fixed/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_fixed/Conv2D?
/conv2d_fixed/batch_normalization/ReadVariableOpReadVariableOp8conv2d_fixed_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_fixed/batch_normalization/ReadVariableOp?
1conv2d_fixed/batch_normalization/ReadVariableOp_1ReadVariableOp:conv2d_fixed_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype023
1conv2d_fixed/batch_normalization/ReadVariableOp_1?
@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpIconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
1conv2d_fixed/batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_fixed/Conv2D:output:07conv2d_fixed/batch_normalization/ReadVariableOp:value:09conv2d_fixed/batch_normalization/ReadVariableOp_1:value:0Hconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<23
1conv2d_fixed/batch_normalization/FusedBatchNormV3?
/conv2d_fixed/batch_normalization/AssignNewValueAssignVariableOpIconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_resource>conv2d_fixed/batch_normalization/FusedBatchNormV3:batch_mean:0A^conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*\
_classR
PNloc:@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype021
/conv2d_fixed/batch_normalization/AssignNewValue?
1conv2d_fixed/batch_normalization/AssignNewValue_1AssignVariableOpKconv2d_fixed_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceBconv2d_fixed/batch_normalization/FusedBatchNormV3:batch_variance:0C^conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*^
_classT
RPloc:@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype023
1conv2d_fixed/batch_normalization/AssignNewValue_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV25conv2d_fixed/batch_normalization/FusedBatchNormV3:y:0inputs concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
&conv2d_na/conv2d/Conv2D/ReadVariableOpReadVariableOp/conv2d_na_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&conv2d_na/conv2d/Conv2D/ReadVariableOp?
conv2d_na/conv2d/Conv2DConv2Dconcatenate/concat:output:0.conv2d_na/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_na/conv2d/Conv2D?
'conv2d_na/conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_na_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_na/conv2d/BiasAdd/ReadVariableOp?
conv2d_na/conv2d/BiasAddBiasAdd conv2d_na/conv2d/Conv2D:output:0/conv2d_na/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_na/conv2d/BiasAdd?
.conv2d_na/batch_normalization_1/ReadVariableOpReadVariableOp7conv2d_na_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype020
.conv2d_na/batch_normalization_1/ReadVariableOp?
0conv2d_na/batch_normalization_1/ReadVariableOp_1ReadVariableOp9conv2d_na_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype022
0conv2d_na/batch_normalization_1/ReadVariableOp_1?
?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
0conv2d_na/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!conv2d_na/conv2d/BiasAdd:output:06conv2d_na/batch_normalization_1/ReadVariableOp:value:08conv2d_na/batch_normalization_1/ReadVariableOp_1:value:0Gconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Iconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<22
0conv2d_na/batch_normalization_1/FusedBatchNormV3?
.conv2d_na/batch_normalization_1/AssignNewValueAssignVariableOpHconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_resource=conv2d_na/batch_normalization_1/FusedBatchNormV3:batch_mean:0@^conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*[
_classQ
OMloc:@conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype020
.conv2d_na/batch_normalization_1/AssignNewValue?
0conv2d_na/batch_normalization_1/AssignNewValue_1AssignVariableOpJconv2d_na_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceAconv2d_na/batch_normalization_1/FusedBatchNormV3:batch_variance:0B^conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*]
_classS
QOloc:@conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype022
0conv2d_na/batch_normalization_1/AssignNewValue_1?
conv2d_na/add/ReadVariableOpReadVariableOp0conv2d_na_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d_na/add/ReadVariableOp?
conv2d_na/addAddV24conv2d_na/batch_normalization_1/FusedBatchNormV3:y:0$conv2d_na/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_na/addw
conv2d_na/ReluReluconv2d_na/add:z:0*
T0*1
_output_shapes
:???????????2
conv2d_na/Relu?
$conv2d_fixed_1/Conv2D/ReadVariableOpReadVariableOp-conv2d_fixed_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$conv2d_fixed_1/Conv2D/ReadVariableOp?
conv2d_fixed_1/Conv2DConv2Dconv2d_na/Relu:activations:0,conv2d_fixed_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_fixed_1/Conv2D?
3conv2d_fixed_1/batch_normalization_2/ReadVariableOpReadVariableOp<conv2d_fixed_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype025
3conv2d_fixed_1/batch_normalization_2/ReadVariableOp?
5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp>conv2d_fixed_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype027
5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1?
Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpMconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
5conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_fixed_1/Conv2D:output:0;conv2d_fixed_1/batch_normalization_2/ReadVariableOp:value:0=conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1:value:0Lconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Nconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z?:::::*
epsilon%o?:*
exponential_avg_factor%
?#<27
5conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3?
3conv2d_fixed_1/batch_normalization_2/AssignNewValueAssignVariableOpMconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceBconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3:batch_mean:0E^conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*`
_classV
TRloc:@conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype025
3conv2d_fixed_1/batch_normalization_2/AssignNewValue?
5conv2d_fixed_1/batch_normalization_2/AssignNewValue_1AssignVariableOpOconv2d_fixed_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceFconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3:batch_variance:0G^conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*b
_classX
VTloc:@conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype027
5conv2d_fixed_1/batch_normalization_2/AssignNewValue_1?
tf.image.resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Z   ?   2
tf.image.resize/resize/size?
%tf.image.resize/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0$tf.image.resize/resize/size:output:0*
T0*0
_output_shapes
:?????????Z?*
half_pixel_centers(2'
%tf.image.resize/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV29conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3:y:06tf.image.resize/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?2
concatenate_1/concat?
*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp?
conv2d_na_1/conv2d_1/Conv2DConv2Dconcatenate_1/concat:output:02conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_na_1/conv2d_1/Conv2D?
+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp?
conv2d_na_1/conv2d_1/BiasAddBiasAdd$conv2d_na_1/conv2d_1/Conv2D:output:03conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_1/conv2d_1/BiasAdd?
0conv2d_na_1/batch_normalization_3/ReadVariableOpReadVariableOp9conv2d_na_1_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_1/batch_normalization_3/ReadVariableOp?
2conv2d_na_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp;conv2d_na_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_1/batch_normalization_3/ReadVariableOp_1?
Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%conv2d_na_1/conv2d_1/BiasAdd:output:08conv2d_na_1/batch_normalization_3/ReadVariableOp:value:0:conv2d_na_1/batch_normalization_3/ReadVariableOp_1:value:0Iconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2conv2d_na_1/batch_normalization_3/FusedBatchNormV3?
0conv2d_na_1/batch_normalization_3/AssignNewValueAssignVariableOpJconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?conv2d_na_1/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*]
_classS
QOloc:@conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0conv2d_na_1/batch_normalization_3/AssignNewValue?
2conv2d_na_1/batch_normalization_3/AssignNewValue_1AssignVariableOpLconv2d_na_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCconv2d_na_1/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2conv2d_na_1/batch_normalization_3/AssignNewValue_1?
conv2d_na_1/add/ReadVariableOpReadVariableOp4conv2d_na_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_1/add/ReadVariableOp?
conv2d_na_1/addAddV26conv2d_na_1/batch_normalization_3/FusedBatchNormV3:y:0&conv2d_na_1/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_1/add|
conv2d_na_1/ReluReluconv2d_na_1/add:z:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_1/Relu?
$conv2d_fixed_2/Conv2D/ReadVariableOpReadVariableOp-conv2d_fixed_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$conv2d_fixed_2/Conv2D/ReadVariableOp?
conv2d_fixed_2/Conv2DConv2Dconv2d_na_1/Relu:activations:0,conv2d_fixed_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_fixed_2/Conv2D?
3conv2d_fixed_2/batch_normalization_4/ReadVariableOpReadVariableOp<conv2d_fixed_2_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype025
3conv2d_fixed_2/batch_normalization_4/ReadVariableOp?
5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1ReadVariableOp>conv2d_fixed_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype027
5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1?
Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpMconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
5conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_fixed_2/Conv2D:output:0;conv2d_fixed_2/batch_normalization_4/ReadVariableOp:value:0=conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1:value:0Lconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Nconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<27
5conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3?
3conv2d_fixed_2/batch_normalization_4/AssignNewValueAssignVariableOpMconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceBconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3:batch_mean:0E^conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*`
_classV
TRloc:@conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype025
3conv2d_fixed_2/batch_normalization_4/AssignNewValue?
5conv2d_fixed_2/batch_normalization_4/AssignNewValue_1AssignVariableOpOconv2d_fixed_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceFconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3:batch_variance:0G^conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*b
_classX
VTloc:@conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype027
5conv2d_fixed_2/batch_normalization_4/AssignNewValue_1?
tf.image.resize_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"-   P   2
tf.image.resize_1/resize/size?
'tf.image.resize_1/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0&tf.image.resize_1/resize/size:output:0*
T0*/
_output_shapes
:?????????-P*
half_pixel_centers(2)
'tf.image.resize_1/resize/ResizeBilinearx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV29conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3:y:08tf.image.resize_1/resize/ResizeBilinear:resized_images:0"concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P&2
concatenate_2/concat?
*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02,
*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp?
conv2d_na_2/conv2d_2/Conv2DConv2Dconcatenate_2/concat:output:02conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_na_2/conv2d_2/Conv2D?
+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp?
conv2d_na_2/conv2d_2/BiasAddBiasAdd$conv2d_na_2/conv2d_2/Conv2D:output:03conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_2/conv2d_2/BiasAdd?
0conv2d_na_2/batch_normalization_5/ReadVariableOpReadVariableOp9conv2d_na_2_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_2/batch_normalization_5/ReadVariableOp?
2conv2d_na_2/batch_normalization_5/ReadVariableOp_1ReadVariableOp;conv2d_na_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_2/batch_normalization_5/ReadVariableOp_1?
Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%conv2d_na_2/conv2d_2/BiasAdd:output:08conv2d_na_2/batch_normalization_5/ReadVariableOp:value:0:conv2d_na_2/batch_normalization_5/ReadVariableOp_1:value:0Iconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2conv2d_na_2/batch_normalization_5/FusedBatchNormV3?
0conv2d_na_2/batch_normalization_5/AssignNewValueAssignVariableOpJconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?conv2d_na_2/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*]
_classS
QOloc:@conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0conv2d_na_2/batch_normalization_5/AssignNewValue?
2conv2d_na_2/batch_normalization_5/AssignNewValue_1AssignVariableOpLconv2d_na_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCconv2d_na_2/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2conv2d_na_2/batch_normalization_5/AssignNewValue_1?
conv2d_na_2/add/ReadVariableOpReadVariableOp4conv2d_na_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_2/add/ReadVariableOp?
conv2d_na_2/addAddV26conv2d_na_2/batch_normalization_5/FusedBatchNormV3:y:0&conv2d_na_2/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_2/add{
conv2d_na_2/ReluReluconv2d_na_2/add:z:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_2/Relu?
*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp?
conv2d_na_3/conv2d_3/Conv2DConv2Dconv2d_na_2/Relu:activations:02conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@*
paddingSAME*
strides
2
conv2d_na_3/conv2d_3/Conv2D?
+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp?
conv2d_na_3/conv2d_3/BiasAddBiasAdd$conv2d_na_3/conv2d_3/Conv2D:output:03conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_na_3/conv2d_3/BiasAdd?
0conv2d_na_3/batch_normalization_6/ReadVariableOpReadVariableOp9conv2d_na_3_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_na_3/batch_normalization_6/ReadVariableOp?
2conv2d_na_3/batch_normalization_6/ReadVariableOp_1ReadVariableOp;conv2d_na_3_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2conv2d_na_3/batch_normalization_6/ReadVariableOp_1?
Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_3/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%conv2d_na_3/conv2d_3/BiasAdd:output:08conv2d_na_3/batch_normalization_6/ReadVariableOp:value:0:conv2d_na_3/batch_normalization_6/ReadVariableOp_1:value:0Iconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2conv2d_na_3/batch_normalization_6/FusedBatchNormV3?
0conv2d_na_3/batch_normalization_6/AssignNewValueAssignVariableOpJconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_resource?conv2d_na_3/batch_normalization_6/FusedBatchNormV3:batch_mean:0B^conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*]
_classS
QOloc:@conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0conv2d_na_3/batch_normalization_6/AssignNewValue?
2conv2d_na_3/batch_normalization_6/AssignNewValue_1AssignVariableOpLconv2d_na_3_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceCconv2d_na_3/batch_normalization_6/FusedBatchNormV3:batch_variance:0D^conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2conv2d_na_3/batch_normalization_6/AssignNewValue_1?
conv2d_na_3/add/ReadVariableOpReadVariableOp4conv2d_na_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv2d_na_3/add/ReadVariableOp?
conv2d_na_3/addAddV26conv2d_na_3/batch_normalization_6/FusedBatchNormV3:y:0&conv2d_na_3/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_na_3/add{
conv2d_na_3/ReluReluconv2d_na_3/add:z:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_na_3/Relux
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2conv2d_na_2/Relu:activations:0conv2d_na_3/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P`2
concatenate_3/concat?
*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02,
*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp?
conv2d_na_4/conv2d_4/Conv2DConv2Dconcatenate_3/concat:output:02conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_na_4/conv2d_4/Conv2D?
+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp?
conv2d_na_4/conv2d_4/BiasAddBiasAdd$conv2d_na_4/conv2d_4/Conv2D:output:03conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_4/conv2d_4/BiasAdd?
0conv2d_na_4/batch_normalization_7/ReadVariableOpReadVariableOp9conv2d_na_4_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_4/batch_normalization_7/ReadVariableOp?
2conv2d_na_4/batch_normalization_7/ReadVariableOp_1ReadVariableOp;conv2d_na_4_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_4/batch_normalization_7/ReadVariableOp_1?
Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_4/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%conv2d_na_4/conv2d_4/BiasAdd:output:08conv2d_na_4/batch_normalization_7/ReadVariableOp:value:0:conv2d_na_4/batch_normalization_7/ReadVariableOp_1:value:0Iconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2conv2d_na_4/batch_normalization_7/FusedBatchNormV3?
0conv2d_na_4/batch_normalization_7/AssignNewValueAssignVariableOpJconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_resource?conv2d_na_4/batch_normalization_7/FusedBatchNormV3:batch_mean:0B^conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*]
_classS
QOloc:@conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0conv2d_na_4/batch_normalization_7/AssignNewValue?
2conv2d_na_4/batch_normalization_7/AssignNewValue_1AssignVariableOpLconv2d_na_4_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceCconv2d_na_4/batch_normalization_7/FusedBatchNormV3:batch_variance:0D^conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2conv2d_na_4/batch_normalization_7/AssignNewValue_1?
conv2d_na_4/add/ReadVariableOpReadVariableOp4conv2d_na_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_4/add/ReadVariableOp?
conv2d_na_4/addAddV26conv2d_na_4/batch_normalization_7/FusedBatchNormV3:y:0&conv2d_na_4/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_4/add{
conv2d_na_4/ReluReluconv2d_na_4/add:z:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_na_4/Relu?
4conv2d_fixed__transpose/conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   Z   ?       26
4conv2d_fixed__transpose/conv2d_transpose/input_sizes?
7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOpReadVariableOp@conv2d_fixed__transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype029
7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp?
(conv2d_fixed__transpose/conv2d_transposeConv2DBackpropInput=conv2d_fixed__transpose/conv2d_transpose/input_sizes:output:0?conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_na_4/Relu:activations:0*
T0*0
_output_shapes
:?????????Z? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2*
(conv2d_fixed__transpose/conv2d_transpose?
<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOpReadVariableOpEconv2d_fixed__transpose_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02>
<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?
>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1ReadVariableOpGconv2d_fixed__transpose_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1?
Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpVconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02O
Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
>conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3FusedBatchNormV31conv2d_fixed__transpose/conv2d_transpose:output:0Dconv2d_fixed__transpose/batch_normalization_8/ReadVariableOp:value:0Fconv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1:value:0Uconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Wconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2@
>conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3?
<conv2d_fixed__transpose/batch_normalization_8/AssignNewValueAssignVariableOpVconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceKconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3:batch_mean:0N^conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*i
_class_
][loc:@conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02>
<conv2d_fixed__transpose/batch_normalization_8/AssignNewValue?
>conv2d_fixed__transpose/batch_normalization_8/AssignNewValue_1AssignVariableOpXconv2d_fixed__transpose_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceOconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3:batch_variance:0P^conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*k
_classa
_]loc:@conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02@
>conv2d_fixed__transpose/batch_normalization_8/AssignNewValue_1x
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2conv2d_na_1/Relu:activations:0Bconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3:y:0"concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?@2
concatenate_4/concat?
*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3conv2d_na_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02,
*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp?
conv2d_na_5/conv2d_5/Conv2DConv2Dconcatenate_4/concat:output:02conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_na_5/conv2d_5/Conv2D?
+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4conv2d_na_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp?
conv2d_na_5/conv2d_5/BiasAddBiasAdd$conv2d_na_5/conv2d_5/Conv2D:output:03conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_5/conv2d_5/BiasAdd?
0conv2d_na_5/batch_normalization_9/ReadVariableOpReadVariableOp9conv2d_na_5_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_na_5/batch_normalization_9/ReadVariableOp?
2conv2d_na_5/batch_normalization_9/ReadVariableOp_1ReadVariableOp;conv2d_na_5_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype024
2conv2d_na_5/batch_normalization_9/ReadVariableOp_1?
Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpJconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
2conv2d_na_5/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%conv2d_na_5/conv2d_5/BiasAdd:output:08conv2d_na_5/batch_normalization_9/ReadVariableOp:value:0:conv2d_na_5/batch_normalization_9/ReadVariableOp_1:value:0Iconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Kconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2conv2d_na_5/batch_normalization_9/FusedBatchNormV3?
0conv2d_na_5/batch_normalization_9/AssignNewValueAssignVariableOpJconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_resource?conv2d_na_5/batch_normalization_9/FusedBatchNormV3:batch_mean:0B^conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*]
_classS
QOloc:@conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0conv2d_na_5/batch_normalization_9/AssignNewValue?
2conv2d_na_5/batch_normalization_9/AssignNewValue_1AssignVariableOpLconv2d_na_5_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceCconv2d_na_5/batch_normalization_9/FusedBatchNormV3:batch_variance:0D^conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2conv2d_na_5/batch_normalization_9/AssignNewValue_1?
conv2d_na_5/add/ReadVariableOpReadVariableOp4conv2d_na_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv2d_na_5/add/ReadVariableOp?
conv2d_na_5/addAddV26conv2d_na_5/batch_normalization_9/FusedBatchNormV3:y:0&conv2d_na_5/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_5/add|
conv2d_na_5/ReluReluconv2d_na_5/add:z:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_na_5/Relu?
6conv2d_fixed__transpose_1/conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   ?   @      28
6conv2d_fixed__transpose_1/conv2d_transpose/input_sizes?
9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpBconv2d_fixed__transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02;
9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp?
*conv2d_fixed__transpose_1/conv2d_transposeConv2DBackpropInput?conv2d_fixed__transpose_1/conv2d_transpose/input_sizes:output:0Aconv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp:value:0conv2d_na_5/Relu:activations:0*
T0*1
_output_shapes
:??????????? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2,
*conv2d_fixed__transpose_1/conv2d_transpose?
?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOpReadVariableOpHconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02A
?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?
Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1ReadVariableOpJconv2d_fixed__transpose_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02C
Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1?
Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYconv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02R
Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02T
Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
Aconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV33conv2d_fixed__transpose_1/conv2d_transpose:output:0Gconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp:value:0Iconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1:value:0Xconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Zconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2C
Aconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3?
?conv2d_fixed__transpose_1/batch_normalization_10/AssignNewValueAssignVariableOpYconv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceNconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3:batch_mean:0Q^conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*l
_classb
`^loc:@conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02A
?conv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue?
Aconv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue_1AssignVariableOp[conv2d_fixed__transpose_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceRconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3:batch_variance:0S^conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*n
_classd
b`loc:@conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02C
Aconv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue_1x
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2conv2d_na/Relu:activations:0Econv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3:y:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????(2
concatenate_5/concat?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dconcatenate_5/concat:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_6/BiasAdd?
conv2d_6/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv2d_6/Max/reduction_indices?
conv2d_6/MaxMaxconv2d_6/BiasAdd:output:0'conv2d_6/Max/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
conv2d_6/Max?
conv2d_6/subSubconv2d_6/BiasAdd:output:0conv2d_6/Max:output:0*
T0*1
_output_shapes
:???????????2
conv2d_6/subq
conv2d_6/ExpExpconv2d_6/sub:z:0*
T0*1
_output_shapes
:???????????2
conv2d_6/Exp?
conv2d_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv2d_6/Sum/reduction_indices?
conv2d_6/SumSumconv2d_6/Exp:y:0'conv2d_6/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
conv2d_6/Sum?
conv2d_6/truedivRealDivconv2d_6/Exp:y:0conv2d_6/Sum:output:0*
T0*1
_output_shapes
:???????????2
conv2d_6/truediv?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/conv2d_na_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3conv2d_na_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?+
IdentityIdentityconv2d_6/truediv:z:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp#^conv2d_fixed/Conv2D/ReadVariableOp0^conv2d_fixed/batch_normalization/AssignNewValue2^conv2d_fixed/batch_normalization/AssignNewValue_1A^conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOpC^conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_10^conv2d_fixed/batch_normalization/ReadVariableOp2^conv2d_fixed/batch_normalization/ReadVariableOp_1%^conv2d_fixed_1/Conv2D/ReadVariableOp4^conv2d_fixed_1/batch_normalization_2/AssignNewValue6^conv2d_fixed_1/batch_normalization_2/AssignNewValue_1E^conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpG^conv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_14^conv2d_fixed_1/batch_normalization_2/ReadVariableOp6^conv2d_fixed_1/batch_normalization_2/ReadVariableOp_1%^conv2d_fixed_2/Conv2D/ReadVariableOp4^conv2d_fixed_2/batch_normalization_4/AssignNewValue6^conv2d_fixed_2/batch_normalization_4/AssignNewValue_1E^conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpG^conv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_14^conv2d_fixed_2/batch_normalization_4/ReadVariableOp6^conv2d_fixed_2/batch_normalization_4/ReadVariableOp_1=^conv2d_fixed__transpose/batch_normalization_8/AssignNewValue?^conv2d_fixed__transpose/batch_normalization_8/AssignNewValue_1N^conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpP^conv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=^conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp?^conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_18^conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp@^conv2d_fixed__transpose_1/batch_normalization_10/AssignNewValueB^conv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue_1Q^conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpS^conv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@^conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOpB^conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1:^conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp^conv2d_na/add/ReadVariableOp/^conv2d_na/batch_normalization_1/AssignNewValue1^conv2d_na/batch_normalization_1/AssignNewValue_1@^conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOpB^conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/^conv2d_na/batch_normalization_1/ReadVariableOp1^conv2d_na/batch_normalization_1/ReadVariableOp_1(^conv2d_na/conv2d/BiasAdd/ReadVariableOp'^conv2d_na/conv2d/Conv2D/ReadVariableOp:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_1/add/ReadVariableOp1^conv2d_na_1/batch_normalization_3/AssignNewValue3^conv2d_na_1/batch_normalization_3/AssignNewValue_1B^conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^conv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_1/batch_normalization_3/ReadVariableOp3^conv2d_na_1/batch_normalization_3/ReadVariableOp_1,^conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp+^conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_2/add/ReadVariableOp1^conv2d_na_2/batch_normalization_5/AssignNewValue3^conv2d_na_2/batch_normalization_5/AssignNewValue_1B^conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^conv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_2/batch_normalization_5/ReadVariableOp3^conv2d_na_2/batch_normalization_5/ReadVariableOp_1,^conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp+^conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_3/add/ReadVariableOp1^conv2d_na_3/batch_normalization_6/AssignNewValue3^conv2d_na_3/batch_normalization_6/AssignNewValue_1B^conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^conv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_3/batch_normalization_6/ReadVariableOp3^conv2d_na_3/batch_normalization_6/ReadVariableOp_1,^conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp+^conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_4/add/ReadVariableOp1^conv2d_na_4/batch_normalization_7/AssignNewValue3^conv2d_na_4/batch_normalization_7/AssignNewValue_1B^conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^conv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_4/batch_normalization_7/ReadVariableOp3^conv2d_na_4/batch_normalization_7/ReadVariableOp_1,^conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp+^conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp^conv2d_na_5/add/ReadVariableOp1^conv2d_na_5/batch_normalization_9/AssignNewValue3^conv2d_na_5/batch_normalization_9/AssignNewValue_1B^conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpD^conv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_11^conv2d_na_5/batch_normalization_9/ReadVariableOp3^conv2d_na_5/batch_normalization_9/ReadVariableOp_1,^conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp+^conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2H
"conv2d_fixed/Conv2D/ReadVariableOp"conv2d_fixed/Conv2D/ReadVariableOp2b
/conv2d_fixed/batch_normalization/AssignNewValue/conv2d_fixed/batch_normalization/AssignNewValue2f
1conv2d_fixed/batch_normalization/AssignNewValue_11conv2d_fixed/batch_normalization/AssignNewValue_12?
@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp@conv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Bconv2d_fixed/batch_normalization/FusedBatchNormV3/ReadVariableOp_12b
/conv2d_fixed/batch_normalization/ReadVariableOp/conv2d_fixed/batch_normalization/ReadVariableOp2f
1conv2d_fixed/batch_normalization/ReadVariableOp_11conv2d_fixed/batch_normalization/ReadVariableOp_12L
$conv2d_fixed_1/Conv2D/ReadVariableOp$conv2d_fixed_1/Conv2D/ReadVariableOp2j
3conv2d_fixed_1/batch_normalization_2/AssignNewValue3conv2d_fixed_1/batch_normalization_2/AssignNewValue2n
5conv2d_fixed_1/batch_normalization_2/AssignNewValue_15conv2d_fixed_1/batch_normalization_2/AssignNewValue_12?
Dconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpDconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Fconv2d_fixed_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12j
3conv2d_fixed_1/batch_normalization_2/ReadVariableOp3conv2d_fixed_1/batch_normalization_2/ReadVariableOp2n
5conv2d_fixed_1/batch_normalization_2/ReadVariableOp_15conv2d_fixed_1/batch_normalization_2/ReadVariableOp_12L
$conv2d_fixed_2/Conv2D/ReadVariableOp$conv2d_fixed_2/Conv2D/ReadVariableOp2j
3conv2d_fixed_2/batch_normalization_4/AssignNewValue3conv2d_fixed_2/batch_normalization_4/AssignNewValue2n
5conv2d_fixed_2/batch_normalization_4/AssignNewValue_15conv2d_fixed_2/batch_normalization_4/AssignNewValue_12?
Dconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpDconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Fconv2d_fixed_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12j
3conv2d_fixed_2/batch_normalization_4/ReadVariableOp3conv2d_fixed_2/batch_normalization_4/ReadVariableOp2n
5conv2d_fixed_2/batch_normalization_4/ReadVariableOp_15conv2d_fixed_2/batch_normalization_4/ReadVariableOp_12|
<conv2d_fixed__transpose/batch_normalization_8/AssignNewValue<conv2d_fixed__transpose/batch_normalization_8/AssignNewValue2?
>conv2d_fixed__transpose/batch_normalization_8/AssignNewValue_1>conv2d_fixed__transpose/batch_normalization_8/AssignNewValue_12?
Mconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOpMconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Oconv2d_fixed__transpose/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12|
<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp<conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp2?
>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_1>conv2d_fixed__transpose/batch_normalization_8/ReadVariableOp_12r
7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp7conv2d_fixed__transpose/conv2d_transpose/ReadVariableOp2?
?conv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue?conv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue2?
Aconv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue_1Aconv2d_fixed__transpose_1/batch_normalization_10/AssignNewValue_12?
Pconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpPconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Rconv2d_fixed__transpose_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12?
?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp?conv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp2?
Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_1Aconv2d_fixed__transpose_1/batch_normalization_10/ReadVariableOp_12v
9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp9conv2d_fixed__transpose_1/conv2d_transpose/ReadVariableOp2<
conv2d_na/add/ReadVariableOpconv2d_na/add/ReadVariableOp2`
.conv2d_na/batch_normalization_1/AssignNewValue.conv2d_na/batch_normalization_1/AssignNewValue2d
0conv2d_na/batch_normalization_1/AssignNewValue_10conv2d_na/batch_normalization_1/AssignNewValue_12?
?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?conv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Aconv2d_na/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12`
.conv2d_na/batch_normalization_1/ReadVariableOp.conv2d_na/batch_normalization_1/ReadVariableOp2d
0conv2d_na/batch_normalization_1/ReadVariableOp_10conv2d_na/batch_normalization_1/ReadVariableOp_12R
'conv2d_na/conv2d/BiasAdd/ReadVariableOp'conv2d_na/conv2d/BiasAdd/ReadVariableOp2P
&conv2d_na/conv2d/Conv2D/ReadVariableOp&conv2d_na/conv2d/Conv2D/ReadVariableOp2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_1/add/ReadVariableOpconv2d_na_1/add/ReadVariableOp2d
0conv2d_na_1/batch_normalization_3/AssignNewValue0conv2d_na_1/batch_normalization_3/AssignNewValue2h
2conv2d_na_1/batch_normalization_3/AssignNewValue_12conv2d_na_1/batch_normalization_3/AssignNewValue_12?
Aconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_1/batch_normalization_3/ReadVariableOp0conv2d_na_1/batch_normalization_3/ReadVariableOp2h
2conv2d_na_1/batch_normalization_3/ReadVariableOp_12conv2d_na_1/batch_normalization_3/ReadVariableOp_12Z
+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp+conv2d_na_1/conv2d_1/BiasAdd/ReadVariableOp2X
*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp*conv2d_na_1/conv2d_1/Conv2D/ReadVariableOp2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_2/add/ReadVariableOpconv2d_na_2/add/ReadVariableOp2d
0conv2d_na_2/batch_normalization_5/AssignNewValue0conv2d_na_2/batch_normalization_5/AssignNewValue2h
2conv2d_na_2/batch_normalization_5/AssignNewValue_12conv2d_na_2/batch_normalization_5/AssignNewValue_12?
Aconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_2/batch_normalization_5/ReadVariableOp0conv2d_na_2/batch_normalization_5/ReadVariableOp2h
2conv2d_na_2/batch_normalization_5/ReadVariableOp_12conv2d_na_2/batch_normalization_5/ReadVariableOp_12Z
+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp+conv2d_na_2/conv2d_2/BiasAdd/ReadVariableOp2X
*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp*conv2d_na_2/conv2d_2/Conv2D/ReadVariableOp2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_3/add/ReadVariableOpconv2d_na_3/add/ReadVariableOp2d
0conv2d_na_3/batch_normalization_6/AssignNewValue0conv2d_na_3/batch_normalization_6/AssignNewValue2h
2conv2d_na_3/batch_normalization_6/AssignNewValue_12conv2d_na_3/batch_normalization_6/AssignNewValue_12?
Aconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_3/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_3/batch_normalization_6/ReadVariableOp0conv2d_na_3/batch_normalization_6/ReadVariableOp2h
2conv2d_na_3/batch_normalization_6/ReadVariableOp_12conv2d_na_3/batch_normalization_6/ReadVariableOp_12Z
+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp+conv2d_na_3/conv2d_3/BiasAdd/ReadVariableOp2X
*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp*conv2d_na_3/conv2d_3/Conv2D/ReadVariableOp2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_4/add/ReadVariableOpconv2d_na_4/add/ReadVariableOp2d
0conv2d_na_4/batch_normalization_7/AssignNewValue0conv2d_na_4/batch_normalization_7/AssignNewValue2h
2conv2d_na_4/batch_normalization_7/AssignNewValue_12conv2d_na_4/batch_normalization_7/AssignNewValue_12?
Aconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_4/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_4/batch_normalization_7/ReadVariableOp0conv2d_na_4/batch_normalization_7/ReadVariableOp2h
2conv2d_na_4/batch_normalization_7/ReadVariableOp_12conv2d_na_4/batch_normalization_7/ReadVariableOp_12Z
+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp+conv2d_na_4/conv2d_4/BiasAdd/ReadVariableOp2X
*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp*conv2d_na_4/conv2d_4/Conv2D/ReadVariableOp2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2@
conv2d_na_5/add/ReadVariableOpconv2d_na_5/add/ReadVariableOp2d
0conv2d_na_5/batch_normalization_9/AssignNewValue0conv2d_na_5/batch_normalization_9/AssignNewValue2h
2conv2d_na_5/batch_normalization_9/AssignNewValue_12conv2d_na_5/batch_normalization_9/AssignNewValue_12?
Aconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOpAconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Cconv2d_na_5/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12d
0conv2d_na_5/batch_normalization_9/ReadVariableOp0conv2d_na_5/batch_normalization_9/ReadVariableOp2h
2conv2d_na_5/batch_normalization_9/ReadVariableOp_12conv2d_na_5/batch_normalization_9/ReadVariableOp_12Z
+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp+conv2d_na_5/conv2d_5/BiasAdd/ReadVariableOp2X
*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp*conv2d_na_5/conv2d_5/Conv2D/ReadVariableOp2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_2226

inputs"
conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_6_7300J
Fconv2d_na_5_conv2d_5_kernel_regularizer_square_readvariableop_resource
identity??=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconv2d_na_5_conv2d_5_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
IdentityIdentity/conv2d_na_5/conv2d_5/kernel/Regularizer/mul:z:0>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp
?
?
4__inference_batch_normalization_2_layer_call_fn_6754

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_9_layer_call_fn_7276

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6723

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7037

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
s
G__inference_concatenate_5_layer_call_and_return_conditional_losses_6503
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????(2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????(2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:??????????? :[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6811

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6886

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
8__inference_conv2d_fixed__transpose_1_layer_call_fn_6496

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_33642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1096

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?,
?
C__inference_conv2d_na_layer_call_and_return_conditional_losses_2353

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
addY
ReluReluadd:z:0*
T0*1
_output_shapes
:???????????2
Relu?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:???????????::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
s
G__inference_concatenate_3_layer_call_and_return_conditional_losses_6109
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P`2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????-P`2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????-P :?????????-P@:Y U
/
_output_shapes
:?????????-P 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????-P@
"
_user_specified_name
inputs/1
?
?
4__inference_batch_normalization_8_layer_call_fn_7206

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_19282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_6_layer_call_fn_7050

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_17202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?,
?
C__inference_conv2d_na_layer_call_and_return_conditional_losses_5570

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
addY
ReluReluadd:z:0*
T0*1
_output_shapes
:???????????2
Relu?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:???????????::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_5_layer_call_fn_6403

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_32692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?@::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?@
 
_user_specified_nameinputs
?#
?
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_2647

inputs"
conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_1:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_4_layer_call_fn_6899

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_15122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_7_layer_call_fn_7131

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_4221

inputs
conv2d_fixed_4030
conv2d_fixed_4032
conv2d_fixed_4034
conv2d_fixed_4036
conv2d_fixed_4038
conv2d_na_4042
conv2d_na_4044
conv2d_na_4046
conv2d_na_4048
conv2d_na_4050
conv2d_na_4052
conv2d_fixed_1_4055
conv2d_fixed_1_4057
conv2d_fixed_1_4059
conv2d_fixed_1_4061
conv2d_fixed_1_4063
conv2d_na_1_4069
conv2d_na_1_4071
conv2d_na_1_4073
conv2d_na_1_4075
conv2d_na_1_4077
conv2d_na_1_4079
conv2d_fixed_2_4082
conv2d_fixed_2_4084
conv2d_fixed_2_4086
conv2d_fixed_2_4088
conv2d_fixed_2_4090
conv2d_na_2_4096
conv2d_na_2_4098
conv2d_na_2_4100
conv2d_na_2_4102
conv2d_na_2_4104
conv2d_na_2_4106
conv2d_na_3_4109
conv2d_na_3_4111
conv2d_na_3_4113
conv2d_na_3_4115
conv2d_na_3_4117
conv2d_na_3_4119
conv2d_na_4_4123
conv2d_na_4_4125
conv2d_na_4_4127
conv2d_na_4_4129
conv2d_na_4_4131
conv2d_na_4_4133 
conv2d_fixed__transpose_4136 
conv2d_fixed__transpose_4138 
conv2d_fixed__transpose_4140 
conv2d_fixed__transpose_4142 
conv2d_fixed__transpose_4144
conv2d_na_5_4148
conv2d_na_5_4150
conv2d_na_5_4152
conv2d_na_5_4154
conv2d_na_5_4156
conv2d_na_5_4158"
conv2d_fixed__transpose_1_4161"
conv2d_fixed__transpose_1_4163"
conv2d_fixed__transpose_1_4165"
conv2d_fixed__transpose_1_4167"
conv2d_fixed__transpose_1_4169
conv2d_6_4173
conv2d_6_4175
identity?? conv2d_6/StatefulPartitionedCall?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?$conv2d_fixed/StatefulPartitionedCall?&conv2d_fixed_1/StatefulPartitionedCall?&conv2d_fixed_2/StatefulPartitionedCall?/conv2d_fixed__transpose/StatefulPartitionedCall?1conv2d_fixed__transpose_1/StatefulPartitionedCall?!conv2d_na/StatefulPartitionedCall?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_1/StatefulPartitionedCall?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_2/StatefulPartitionedCall?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_3/StatefulPartitionedCall?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_4/StatefulPartitionedCall?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_5/StatefulPartitionedCall?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
$conv2d_fixed/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_fixed_4030conv2d_fixed_4032conv2d_fixed_4034conv2d_fixed_4036conv2d_fixed_4038*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_22262&
$conv2d_fixed/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall-conv2d_fixed/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_22762
concatenate/PartitionedCall?
!conv2d_na/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_na_4042conv2d_na_4044conv2d_na_4046conv2d_na_4048conv2d_na_4050conv2d_na_4052*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_na_layer_call_and_return_conditional_losses_23532#
!conv2d_na/StatefulPartitionedCall?
&conv2d_fixed_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0conv2d_fixed_1_4055conv2d_fixed_1_4057conv2d_fixed_1_4059conv2d_fixed_1_4061conv2d_fixed_1_4063*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_24462(
&conv2d_fixed_1/StatefulPartitionedCall?
tf.image.resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Z   ?   2
tf.image.resize/resize/size?
%tf.image.resize/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0$tf.image.resize/resize/size:output:0*
T0*0
_output_shapes
:?????????Z?*
half_pixel_centers(2'
%tf.image.resize/resize/ResizeBilinear?
concatenate_1/PartitionedCallPartitionedCall/conv2d_fixed_1/StatefulPartitionedCall:output:06tf.image.resize/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_24982
concatenate_1/PartitionedCall?
#conv2d_na_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_na_1_4069conv2d_na_1_4071conv2d_na_1_4073conv2d_na_1_4075conv2d_na_1_4077conv2d_na_1_4079*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_25752%
#conv2d_na_1/StatefulPartitionedCall?
&conv2d_fixed_2/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:0conv2d_fixed_2_4082conv2d_fixed_2_4084conv2d_fixed_2_4086conv2d_fixed_2_4088conv2d_fixed_2_4090*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_26682(
&conv2d_fixed_2/StatefulPartitionedCall?
tf.image.resize_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"-   P   2
tf.image.resize_1/resize/size?
'tf.image.resize_1/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0&tf.image.resize_1/resize/size:output:0*
T0*/
_output_shapes
:?????????-P*
half_pixel_centers(2)
'tf.image.resize_1/resize/ResizeBilinear?
concatenate_2/PartitionedCallPartitionedCall/conv2d_fixed_2/StatefulPartitionedCall:output:08tf.image.resize_1/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P&* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_27202
concatenate_2/PartitionedCall?
#conv2d_na_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_na_2_4096conv2d_na_2_4098conv2d_na_2_4100conv2d_na_2_4102conv2d_na_2_4104conv2d_na_2_4106*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_27972%
#conv2d_na_2/StatefulPartitionedCall?
#conv2d_na_3/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0conv2d_na_3_4109conv2d_na_3_4111conv2d_na_3_4113conv2d_na_3_4115conv2d_na_3_4117conv2d_na_3_4119*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_29142%
#conv2d_na_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0,conv2d_na_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_29702
concatenate_3/PartitionedCall?
#conv2d_na_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_na_4_4123conv2d_na_4_4125conv2d_na_4_4127conv2d_na_4_4129conv2d_na_4_4131conv2d_na_4_4133*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_30472%
#conv2d_na_4/StatefulPartitionedCall?
/conv2d_fixed__transpose/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_4/StatefulPartitionedCall:output:0conv2d_fixed__transpose_4136conv2d_fixed__transpose_4138conv2d_fixed__transpose_4140conv2d_fixed__transpose_4142conv2d_fixed__transpose_4144*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_314221
/conv2d_fixed__transpose/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:08conv2d_fixed__transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_31922
concatenate_4/PartitionedCall?
#conv2d_na_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_na_5_4148conv2d_na_5_4150conv2d_na_5_4152conv2d_na_5_4154conv2d_na_5_4156conv2d_na_5_4158*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_32692%
#conv2d_na_5/StatefulPartitionedCall?
1conv2d_fixed__transpose_1/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_5/StatefulPartitionedCall:output:0conv2d_fixed__transpose_1_4161conv2d_fixed__transpose_1_4163conv2d_fixed__transpose_1_4165conv2d_fixed__transpose_1_4167conv2d_fixed__transpose_1_4169*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_336423
1conv2d_fixed__transpose_1/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0:conv2d_fixed__transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_34142
concatenate_5/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_6_4173conv2d_6_4175*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_34462"
 conv2d_6/StatefulPartitionedCall?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_4042*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_1_4069*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_2_4096*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_3_4109*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_4_4123*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_5_4148*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_4173*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp%^conv2d_fixed/StatefulPartitionedCall'^conv2d_fixed_1/StatefulPartitionedCall'^conv2d_fixed_2/StatefulPartitionedCall0^conv2d_fixed__transpose/StatefulPartitionedCall2^conv2d_fixed__transpose_1/StatefulPartitionedCall"^conv2d_na/StatefulPartitionedCall:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_1/StatefulPartitionedCall>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_2/StatefulPartitionedCall>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_3/StatefulPartitionedCall>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_4/StatefulPartitionedCall>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_5/StatefulPartitionedCall>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2L
$conv2d_fixed/StatefulPartitionedCall$conv2d_fixed/StatefulPartitionedCall2P
&conv2d_fixed_1/StatefulPartitionedCall&conv2d_fixed_1/StatefulPartitionedCall2P
&conv2d_fixed_2/StatefulPartitionedCall&conv2d_fixed_2/StatefulPartitionedCall2b
/conv2d_fixed__transpose/StatefulPartitionedCall/conv2d_fixed__transpose/StatefulPartitionedCall2f
1conv2d_fixed__transpose_1/StatefulPartitionedCall1conv2d_fixed__transpose_1/StatefulPartitionedCall2F
!conv2d_na/StatefulPartitionedCall!conv2d_na/StatefulPartitionedCall2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_1/StatefulPartitionedCall#conv2d_na_1/StatefulPartitionedCall2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_2/StatefulPartitionedCall#conv2d_na_2/StatefulPartitionedCall2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_3/StatefulPartitionedCall#conv2d_na_3/StatefulPartitionedCall2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_4/StatefulPartitionedCall#conv2d_na_4/StatefulPartitionedCall2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_5/StatefulPartitionedCall#conv2d_na_5/StatefulPartitionedCall2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
X
,__inference_concatenate_2_layer_call_fn_5886
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P&* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_27202
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????-P&2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????-P :?????????-P:Y U
/
_output_shapes
:?????????-P 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????-P
"
_user_specified_name
inputs/1
?
s
G__inference_concatenate_2_layer_call_and_return_conditional_losses_5880
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P&2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????-P&2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????-P :?????????-P:Y U
/
_output_shapes
:?????????-P 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????-P
"
_user_specified_name
inputs/1
??
?
?__inference_model_layer_call_and_return_conditional_losses_3896

inputs
conv2d_fixed_3705
conv2d_fixed_3707
conv2d_fixed_3709
conv2d_fixed_3711
conv2d_fixed_3713
conv2d_na_3717
conv2d_na_3719
conv2d_na_3721
conv2d_na_3723
conv2d_na_3725
conv2d_na_3727
conv2d_fixed_1_3730
conv2d_fixed_1_3732
conv2d_fixed_1_3734
conv2d_fixed_1_3736
conv2d_fixed_1_3738
conv2d_na_1_3744
conv2d_na_1_3746
conv2d_na_1_3748
conv2d_na_1_3750
conv2d_na_1_3752
conv2d_na_1_3754
conv2d_fixed_2_3757
conv2d_fixed_2_3759
conv2d_fixed_2_3761
conv2d_fixed_2_3763
conv2d_fixed_2_3765
conv2d_na_2_3771
conv2d_na_2_3773
conv2d_na_2_3775
conv2d_na_2_3777
conv2d_na_2_3779
conv2d_na_2_3781
conv2d_na_3_3784
conv2d_na_3_3786
conv2d_na_3_3788
conv2d_na_3_3790
conv2d_na_3_3792
conv2d_na_3_3794
conv2d_na_4_3798
conv2d_na_4_3800
conv2d_na_4_3802
conv2d_na_4_3804
conv2d_na_4_3806
conv2d_na_4_3808 
conv2d_fixed__transpose_3811 
conv2d_fixed__transpose_3813 
conv2d_fixed__transpose_3815 
conv2d_fixed__transpose_3817 
conv2d_fixed__transpose_3819
conv2d_na_5_3823
conv2d_na_5_3825
conv2d_na_5_3827
conv2d_na_5_3829
conv2d_na_5_3831
conv2d_na_5_3833"
conv2d_fixed__transpose_1_3836"
conv2d_fixed__transpose_1_3838"
conv2d_fixed__transpose_1_3840"
conv2d_fixed__transpose_1_3842"
conv2d_fixed__transpose_1_3844
conv2d_6_3848
conv2d_6_3850
identity?? conv2d_6/StatefulPartitionedCall?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?$conv2d_fixed/StatefulPartitionedCall?&conv2d_fixed_1/StatefulPartitionedCall?&conv2d_fixed_2/StatefulPartitionedCall?/conv2d_fixed__transpose/StatefulPartitionedCall?1conv2d_fixed__transpose_1/StatefulPartitionedCall?!conv2d_na/StatefulPartitionedCall?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_1/StatefulPartitionedCall?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_2/StatefulPartitionedCall?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_3/StatefulPartitionedCall?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_4/StatefulPartitionedCall?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_5/StatefulPartitionedCall?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
$conv2d_fixed/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_fixed_3705conv2d_fixed_3707conv2d_fixed_3709conv2d_fixed_3711conv2d_fixed_3713*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_22052&
$conv2d_fixed/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall-conv2d_fixed/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_22762
concatenate/PartitionedCall?
!conv2d_na/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_na_3717conv2d_na_3719conv2d_na_3721conv2d_na_3723conv2d_na_3725conv2d_na_3727*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_na_layer_call_and_return_conditional_losses_23202#
!conv2d_na/StatefulPartitionedCall?
&conv2d_fixed_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0conv2d_fixed_1_3730conv2d_fixed_1_3732conv2d_fixed_1_3734conv2d_fixed_1_3736conv2d_fixed_1_3738*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_24252(
&conv2d_fixed_1/StatefulPartitionedCall?
tf.image.resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Z   ?   2
tf.image.resize/resize/size?
%tf.image.resize/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0$tf.image.resize/resize/size:output:0*
T0*0
_output_shapes
:?????????Z?*
half_pixel_centers(2'
%tf.image.resize/resize/ResizeBilinear?
concatenate_1/PartitionedCallPartitionedCall/conv2d_fixed_1/StatefulPartitionedCall:output:06tf.image.resize/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_24982
concatenate_1/PartitionedCall?
#conv2d_na_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_na_1_3744conv2d_na_1_3746conv2d_na_1_3748conv2d_na_1_3750conv2d_na_1_3752conv2d_na_1_3754*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_25422%
#conv2d_na_1/StatefulPartitionedCall?
&conv2d_fixed_2/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:0conv2d_fixed_2_3757conv2d_fixed_2_3759conv2d_fixed_2_3761conv2d_fixed_2_3763conv2d_fixed_2_3765*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_26472(
&conv2d_fixed_2/StatefulPartitionedCall?
tf.image.resize_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"-   P   2
tf.image.resize_1/resize/size?
'tf.image.resize_1/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0&tf.image.resize_1/resize/size:output:0*
T0*/
_output_shapes
:?????????-P*
half_pixel_centers(2)
'tf.image.resize_1/resize/ResizeBilinear?
concatenate_2/PartitionedCallPartitionedCall/conv2d_fixed_2/StatefulPartitionedCall:output:08tf.image.resize_1/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P&* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_27202
concatenate_2/PartitionedCall?
#conv2d_na_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_na_2_3771conv2d_na_2_3773conv2d_na_2_3775conv2d_na_2_3777conv2d_na_2_3779conv2d_na_2_3781*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_27642%
#conv2d_na_2/StatefulPartitionedCall?
#conv2d_na_3/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0conv2d_na_3_3784conv2d_na_3_3786conv2d_na_3_3788conv2d_na_3_3790conv2d_na_3_3792conv2d_na_3_3794*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_28812%
#conv2d_na_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0,conv2d_na_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_29702
concatenate_3/PartitionedCall?
#conv2d_na_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_na_4_3798conv2d_na_4_3800conv2d_na_4_3802conv2d_na_4_3804conv2d_na_4_3806conv2d_na_4_3808*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_30142%
#conv2d_na_4/StatefulPartitionedCall?
/conv2d_fixed__transpose/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_4/StatefulPartitionedCall:output:0conv2d_fixed__transpose_3811conv2d_fixed__transpose_3813conv2d_fixed__transpose_3815conv2d_fixed__transpose_3817conv2d_fixed__transpose_3819*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_312021
/conv2d_fixed__transpose/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:08conv2d_fixed__transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_31922
concatenate_4/PartitionedCall?
#conv2d_na_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_na_5_3823conv2d_na_5_3825conv2d_na_5_3827conv2d_na_5_3829conv2d_na_5_3831conv2d_na_5_3833*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_32362%
#conv2d_na_5/StatefulPartitionedCall?
1conv2d_fixed__transpose_1/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_5/StatefulPartitionedCall:output:0conv2d_fixed__transpose_1_3836conv2d_fixed__transpose_1_3838conv2d_fixed__transpose_1_3840conv2d_fixed__transpose_1_3842conv2d_fixed__transpose_1_3844*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_334223
1conv2d_fixed__transpose_1/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0:conv2d_fixed__transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_34142
concatenate_5/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_6_3848conv2d_6_3850*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_34462"
 conv2d_6/StatefulPartitionedCall?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_3717*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_1_3744*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_2_3771*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_3_3784*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_4_3798*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_5_3823*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_3848*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp%^conv2d_fixed/StatefulPartitionedCall'^conv2d_fixed_1/StatefulPartitionedCall'^conv2d_fixed_2/StatefulPartitionedCall0^conv2d_fixed__transpose/StatefulPartitionedCall2^conv2d_fixed__transpose_1/StatefulPartitionedCall"^conv2d_na/StatefulPartitionedCall:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_1/StatefulPartitionedCall>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_2/StatefulPartitionedCall>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_3/StatefulPartitionedCall>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_4/StatefulPartitionedCall>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_5/StatefulPartitionedCall>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2L
$conv2d_fixed/StatefulPartitionedCall$conv2d_fixed/StatefulPartitionedCall2P
&conv2d_fixed_1/StatefulPartitionedCall&conv2d_fixed_1/StatefulPartitionedCall2P
&conv2d_fixed_2/StatefulPartitionedCall&conv2d_fixed_2/StatefulPartitionedCall2b
/conv2d_fixed__transpose/StatefulPartitionedCall/conv2d_fixed__transpose/StatefulPartitionedCall2f
1conv2d_fixed__transpose_1/StatefulPartitionedCall1conv2d_fixed__transpose_1/StatefulPartitionedCall2F
!conv2d_na/StatefulPartitionedCall!conv2d_na/StatefulPartitionedCall2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_1/StatefulPartitionedCall#conv2d_na_1/StatefulPartitionedCall2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_2/StatefulPartitionedCall#conv2d_na_2/StatefulPartitionedCall2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_3/StatefulPartitionedCall#conv2d_na_3/StatefulPartitionedCall2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_4/StatefulPartitionedCall#conv2d_na_4/StatefulPartitionedCall2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_5/StatefulPartitionedCall#conv2d_na_5/StatefulPartitionedCall2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_fixed_2_layer_call_fn_5858

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_26682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_5843

inputs"
conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp6^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_1:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_6035

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_3/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P@2
Relu?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P ::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
|
'__inference_conv2d_6_layer_call_fn_6547

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_34462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????(::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????(
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_6993J
Fconv2d_na_2_conv2d_2_kernel_regularizer_square_readvariableop_resource
identity??=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconv2d_na_2_conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
IdentityIdentity/conv2d_na_2/conv2d_2/kernel/Regularizer/mul:z:0>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp
?
?
4__inference_batch_normalization_1_layer_call_fn_6692

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_10_layer_call_fn_7364

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_21672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7019

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2063

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?%
?
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_3120

inputs,
(conv2d_transpose_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   Z   ?       2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????Z? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1?
IdentityIdentity*batch_normalization_8/FusedBatchNormV3:y:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????-P :::::2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_3_layer_call_fn_6085

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_29142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
X
,__inference_concatenate_3_layer_call_fn_6115
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_29702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????-P`2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????-P :?????????-P@:Y U
/
_output_shapes
:?????????-P 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????-P@
"
_user_specified_name
inputs/1
?-
?
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_6189

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_4/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_7/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P`::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P`
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_3047

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_4/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_7/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P`::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P`
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6868

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_2914

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_3/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P@2
Relu?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P ::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7338

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1959

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_batch_normalization_layer_call_fn_6622

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_11272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
V
*__inference_concatenate_layer_call_fn_5496
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_22762
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_7263

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2167

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_5927

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_2/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_5/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P&::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P&
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_6558>
:conv2d_6_kernel_regularizer_square_readvariableop_resource
identity??1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_6_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
IdentityIdentity#conv2d_6/kernel/Regularizer/mul:z:02^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1855

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_1_layer_call_fn_6679

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6648

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1127

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_7074J
Fconv2d_na_3_conv2d_3_kernel_regularizer_square_readvariableop_resource
identity??=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconv2d_na_3_conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
IdentityIdentity/conv2d_na_3/conv2d_3/kernel/Regularizer/mul:z:0>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp
?
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2032

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_4_layer_call_fn_6206

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_30472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P`::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P`
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_3699
input_1
conv2d_fixed_3508
conv2d_fixed_3510
conv2d_fixed_3512
conv2d_fixed_3514
conv2d_fixed_3516
conv2d_na_3520
conv2d_na_3522
conv2d_na_3524
conv2d_na_3526
conv2d_na_3528
conv2d_na_3530
conv2d_fixed_1_3533
conv2d_fixed_1_3535
conv2d_fixed_1_3537
conv2d_fixed_1_3539
conv2d_fixed_1_3541
conv2d_na_1_3547
conv2d_na_1_3549
conv2d_na_1_3551
conv2d_na_1_3553
conv2d_na_1_3555
conv2d_na_1_3557
conv2d_fixed_2_3560
conv2d_fixed_2_3562
conv2d_fixed_2_3564
conv2d_fixed_2_3566
conv2d_fixed_2_3568
conv2d_na_2_3574
conv2d_na_2_3576
conv2d_na_2_3578
conv2d_na_2_3580
conv2d_na_2_3582
conv2d_na_2_3584
conv2d_na_3_3587
conv2d_na_3_3589
conv2d_na_3_3591
conv2d_na_3_3593
conv2d_na_3_3595
conv2d_na_3_3597
conv2d_na_4_3601
conv2d_na_4_3603
conv2d_na_4_3605
conv2d_na_4_3607
conv2d_na_4_3609
conv2d_na_4_3611 
conv2d_fixed__transpose_3614 
conv2d_fixed__transpose_3616 
conv2d_fixed__transpose_3618 
conv2d_fixed__transpose_3620 
conv2d_fixed__transpose_3622
conv2d_na_5_3626
conv2d_na_5_3628
conv2d_na_5_3630
conv2d_na_5_3632
conv2d_na_5_3634
conv2d_na_5_3636"
conv2d_fixed__transpose_1_3639"
conv2d_fixed__transpose_1_3641"
conv2d_fixed__transpose_1_3643"
conv2d_fixed__transpose_1_3645"
conv2d_fixed__transpose_1_3647
conv2d_6_3651
conv2d_6_3653
identity?? conv2d_6/StatefulPartitionedCall?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?$conv2d_fixed/StatefulPartitionedCall?&conv2d_fixed_1/StatefulPartitionedCall?&conv2d_fixed_2/StatefulPartitionedCall?/conv2d_fixed__transpose/StatefulPartitionedCall?1conv2d_fixed__transpose_1/StatefulPartitionedCall?!conv2d_na/StatefulPartitionedCall?9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_1/StatefulPartitionedCall?=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_2/StatefulPartitionedCall?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_3/StatefulPartitionedCall?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_4/StatefulPartitionedCall?=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?#conv2d_na_5/StatefulPartitionedCall?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
$conv2d_fixed/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_fixed_3508conv2d_fixed_3510conv2d_fixed_3512conv2d_fixed_3514conv2d_fixed_3516*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_22262&
$conv2d_fixed/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall-conv2d_fixed/StatefulPartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_22762
concatenate/PartitionedCall?
!conv2d_na/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_na_3520conv2d_na_3522conv2d_na_3524conv2d_na_3526conv2d_na_3528conv2d_na_3530*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_na_layer_call_and_return_conditional_losses_23532#
!conv2d_na/StatefulPartitionedCall?
&conv2d_fixed_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0conv2d_fixed_1_3533conv2d_fixed_1_3535conv2d_fixed_1_3537conv2d_fixed_1_3539conv2d_fixed_1_3541*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_24462(
&conv2d_fixed_1/StatefulPartitionedCall?
tf.image.resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Z   ?   2
tf.image.resize/resize/size?
%tf.image.resize/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0$tf.image.resize/resize/size:output:0*
T0*0
_output_shapes
:?????????Z?*
half_pixel_centers(2'
%tf.image.resize/resize/ResizeBilinear?
concatenate_1/PartitionedCallPartitionedCall/conv2d_fixed_1/StatefulPartitionedCall:output:06tf.image.resize/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_24982
concatenate_1/PartitionedCall?
#conv2d_na_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_na_1_3547conv2d_na_1_3549conv2d_na_1_3551conv2d_na_1_3553conv2d_na_1_3555conv2d_na_1_3557*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_25752%
#conv2d_na_1/StatefulPartitionedCall?
&conv2d_fixed_2/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:0conv2d_fixed_2_3560conv2d_fixed_2_3562conv2d_fixed_2_3564conv2d_fixed_2_3566conv2d_fixed_2_3568*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_26682(
&conv2d_fixed_2/StatefulPartitionedCall?
tf.image.resize_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"-   P   2
tf.image.resize_1/resize/size?
'tf.image.resize_1/resize/ResizeBilinearResizeBilinear$concatenate/PartitionedCall:output:0&tf.image.resize_1/resize/size:output:0*
T0*/
_output_shapes
:?????????-P*
half_pixel_centers(2)
'tf.image.resize_1/resize/ResizeBilinear?
concatenate_2/PartitionedCallPartitionedCall/conv2d_fixed_2/StatefulPartitionedCall:output:08tf.image.resize_1/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P&* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_27202
concatenate_2/PartitionedCall?
#conv2d_na_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_na_2_3574conv2d_na_2_3576conv2d_na_2_3578conv2d_na_2_3580conv2d_na_2_3582conv2d_na_2_3584*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_27972%
#conv2d_na_2/StatefulPartitionedCall?
#conv2d_na_3/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0conv2d_na_3_3587conv2d_na_3_3589conv2d_na_3_3591conv2d_na_3_3593conv2d_na_3_3595conv2d_na_3_3597*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_29142%
#conv2d_na_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,conv2d_na_2/StatefulPartitionedCall:output:0,conv2d_na_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_29702
concatenate_3/PartitionedCall?
#conv2d_na_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_na_4_3601conv2d_na_4_3603conv2d_na_4_3605conv2d_na_4_3607conv2d_na_4_3609conv2d_na_4_3611*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_30472%
#conv2d_na_4/StatefulPartitionedCall?
/conv2d_fixed__transpose/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_4/StatefulPartitionedCall:output:0conv2d_fixed__transpose_3614conv2d_fixed__transpose_3616conv2d_fixed__transpose_3618conv2d_fixed__transpose_3620conv2d_fixed__transpose_3622*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_314221
/conv2d_fixed__transpose/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall,conv2d_na_1/StatefulPartitionedCall:output:08conv2d_fixed__transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_31922
concatenate_4/PartitionedCall?
#conv2d_na_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_na_5_3626conv2d_na_5_3628conv2d_na_5_3630conv2d_na_5_3632conv2d_na_5_3634conv2d_na_5_3636*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_32692%
#conv2d_na_5/StatefulPartitionedCall?
1conv2d_fixed__transpose_1/StatefulPartitionedCallStatefulPartitionedCall,conv2d_na_5/StatefulPartitionedCall:output:0conv2d_fixed__transpose_1_3639conv2d_fixed__transpose_1_3641conv2d_fixed__transpose_1_3643conv2d_fixed__transpose_1_3645conv2d_fixed__transpose_1_3647*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_336423
1conv2d_fixed__transpose_1/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall*conv2d_na/StatefulPartitionedCall:output:0:conv2d_fixed__transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_34142
concatenate_5/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_6_3651conv2d_6_3653*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_34462"
 conv2d_6/StatefulPartitionedCall?
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_3520*&
_output_shapes
:*
dtype02;
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_na/conv2d/kernel/Regularizer/SquareSquareAconv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2,
*conv2d_na/conv2d/kernel/Regularizer/Square?
)conv2d_na/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_na/conv2d/kernel/Regularizer/Const?
'conv2d_na/conv2d/kernel/Regularizer/SumSum.conv2d_na/conv2d/kernel/Regularizer/Square:y:02conv2d_na/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/Sum?
)conv2d_na/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_na/conv2d/kernel/Regularizer/mul/x?
'conv2d_na/conv2d/kernel/Regularizer/mulMul2conv2d_na/conv2d/kernel/Regularizer/mul/x:output:00conv2d_na/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_na/conv2d/kernel/Regularizer/mul?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_1_3547*&
_output_shapes
: *
dtype02?
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_1/conv2d_1/kernel/Regularizer/SquareSquareEconv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.conv2d_na_1/conv2d_1/kernel/Regularizer/Square?
-conv2d_na_1/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/Const?
+conv2d_na_1/conv2d_1/kernel/Regularizer/SumSum2conv2d_na_1/conv2d_1/kernel/Regularizer/Square:y:06conv2d_na_1/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/Sum?
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x?
+conv2d_na_1/conv2d_1/kernel/Regularizer/mulMul6conv2d_na_1/conv2d_1/kernel/Regularizer/mul/x:output:04conv2d_na_1/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_1/conv2d_1/kernel/Regularizer/mul?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_2_3574*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_3_3587*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_4_3601*&
_output_shapes
:` *
dtype02?
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_4/conv2d_4/kernel/Regularizer/SquareSquareEconv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:` 20
.conv2d_na_4/conv2d_4/kernel/Regularizer/Square?
-conv2d_na_4/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/Const?
+conv2d_na_4/conv2d_4/kernel/Regularizer/SumSum2conv2d_na_4/conv2d_4/kernel/Regularizer/Square:y:06conv2d_na_4/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/Sum?
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x?
+conv2d_na_4/conv2d_4/kernel/Regularizer/mulMul6conv2d_na_4/conv2d_4/kernel/Regularizer/mul/x:output:04conv2d_na_4/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_4/conv2d_4/kernel/Regularizer/mul?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_na_5_3626*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_3651*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp%^conv2d_fixed/StatefulPartitionedCall'^conv2d_fixed_1/StatefulPartitionedCall'^conv2d_fixed_2/StatefulPartitionedCall0^conv2d_fixed__transpose/StatefulPartitionedCall2^conv2d_fixed__transpose_1/StatefulPartitionedCall"^conv2d_na/StatefulPartitionedCall:^conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_1/StatefulPartitionedCall>^conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_2/StatefulPartitionedCall>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_3/StatefulPartitionedCall>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_4/StatefulPartitionedCall>^conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp$^conv2d_na_5/StatefulPartitionedCall>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2L
$conv2d_fixed/StatefulPartitionedCall$conv2d_fixed/StatefulPartitionedCall2P
&conv2d_fixed_1/StatefulPartitionedCall&conv2d_fixed_1/StatefulPartitionedCall2P
&conv2d_fixed_2/StatefulPartitionedCall&conv2d_fixed_2/StatefulPartitionedCall2b
/conv2d_fixed__transpose/StatefulPartitionedCall/conv2d_fixed__transpose/StatefulPartitionedCall2f
1conv2d_fixed__transpose_1/StatefulPartitionedCall1conv2d_fixed__transpose_1/StatefulPartitionedCall2F
!conv2d_na/StatefulPartitionedCall!conv2d_na/StatefulPartitionedCall2v
9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp9conv2d_na/conv2d/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_1/StatefulPartitionedCall#conv2d_na_1/StatefulPartitionedCall2~
=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_1/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_2/StatefulPartitionedCall#conv2d_na_2/StatefulPartitionedCall2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_3/StatefulPartitionedCall#conv2d_na_3/StatefulPartitionedCall2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_4/StatefulPartitionedCall#conv2d_na_4/StatefulPartitionedCall2~
=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_4/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2J
#conv2d_na_5/StatefulPartitionedCall#conv2d_na_5/StatefulPartitionedCall2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?&
?
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_3342

inputs,
(conv2d_transpose_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity??%batch_normalization_10/AssignNewValue?'batch_normalization_10/AssignNewValue_1?6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   ?   @      2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:??????????? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_10/FusedBatchNormV3?
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue?
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1?
IdentityIdentity+batch_normalization_10/FusedBatchNormV3:y:0&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7118

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6793

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_4525
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_10342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?#
?
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_2425

inputs"
conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity??Conv2D/ReadVariableOp?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z?*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z?:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0^Conv2D/ReadVariableOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*0
_output_shapes
:?????????Z?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_3446

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Maxm
subSubBiasAdd:output:0Max:output:0*
T0*1
_output_shapes
:???????????2
subV
ExpExpsub:z:0*
T0*1
_output_shapes
:???????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Sump
truedivRealDivExp:y:0Sum:output:0*
T0*1
_output_shapes
:???????????2	
truediv?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:(2$
"conv2d_6/kernel/Regularizer/Square?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/Const?
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
IdentityIdentitytruediv:z:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????(
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1616

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
q
G__inference_concatenate_2_layer_call_and_return_conditional_losses_2720

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????-P&2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????-P&2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????-P :?????????-P:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????-P
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_10_layer_call_fn_7351

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_21362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2136

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_2_layer_call_fn_5977

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-P *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_27972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P&::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P&
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_1_layer_call_fn_5799

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_25752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5409

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_42212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?-
?
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_6386

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_5/BiasAdd?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3?
add/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_9/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp6^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?@::::::2(
add/ReadVariableOpadd/ReadVariableOp2n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?@
 
_user_specified_nameinputs
?
?
8__inference_conv2d_fixed__transpose_1_layer_call_fn_6481

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_33642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_2881

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
conv2d_3/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P@2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P@2
Relu?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_3/conv2d_3/kernel/Regularizer/SquareSquareEconv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.conv2d_na_3/conv2d_3/kernel/Regularizer/Square?
-conv2d_na_3/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/Const?
+conv2d_na_3/conv2d_3/kernel/Regularizer/SumSum2conv2d_na_3/conv2d_3/kernel/Regularizer/Square:y:06conv2d_na_3/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/Sum?
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x?
+conv2d_na_3/conv2d_3/kernel/Regularizer/mulMul6conv2d_na_3/conv2d_3/kernel/Regularizer/mul/x:output:04conv2d_na_3/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_3/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp>^conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P ::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2~
=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_3/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?
s
G__inference_concatenate_4_layer_call_and_return_conditional_losses_6306
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?@2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????Z?@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????Z? :?????????Z? :Z V
0
_output_shapes
:?????????Z? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????Z? 
"
_user_specified_name
inputs/1
?
q
G__inference_concatenate_4_layer_call_and_return_conditional_losses_3192

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????Z?@2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????Z?@2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????Z? :?????????Z? :X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_4025
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*K
_read_only_resource_inputs-
+)	"#$%()*+./034569:;>?*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_38962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
+__inference_conv2d_fixed_layer_call_fn_5483

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_22262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_na_5_layer_call_fn_6420

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_32692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?@::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????Z?@
 
_user_specified_nameinputs
?
?
6__inference_conv2d_fixed__transpose_layer_call_fn_6284

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????Z? *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_31422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????-P :::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-P 
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_6353

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
conv2d_5/BiasAdd?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????Z? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_9/FusedBatchNormV3?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_9/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Z? 2
addX
ReluReluadd:z:0*
T0*0
_output_shapes
:?????????Z? 2
Relu?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02?
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_5/conv2d_5/kernel/Regularizer/SquareSquareEconv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 20
.conv2d_na_5/conv2d_5/kernel/Regularizer/Square?
-conv2d_na_5/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/Const?
+conv2d_na_5/conv2d_5/kernel/Regularizer/SumSum2conv2d_na_5/conv2d_5/kernel/Regularizer/Square:y:06conv2d_na_5/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/Sum?
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x?
+conv2d_na_5/conv2d_5/kernel/Regularizer/mulMul6conv2d_na_5/conv2d_5/kernel/Regularizer/mul/x:output:04conv2d_na_5/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_5/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp>^conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????Z? 2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????Z?@::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2~
=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_5/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????Z?@
 
_user_specified_nameinputs
?
q
G__inference_concatenate_5_layer_call_and_return_conditional_losses_3414

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????(2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????(2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:??????????? :Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1720

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1304

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_3364

inputs,
(conv2d_transpose_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity??6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?conv2d_transpose/ReadVariableOp?
conv2d_transpose/input_sizesConst*
_output_shapes
:*
dtype0*%
valueB"   ?   @      2
conv2d_transpose/input_sizes?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:??????????? *!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_transpose:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3?
IdentityIdentity+batch_normalization_10/FusedBatchNormV3:y:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1 ^conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????Z? :::::2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????Z? 
 
_user_specified_nameinputs
?7
?
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_2764

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??add/ReadVariableOp?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
conv2d_2/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????-P : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
add/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
add/ReadVariableOp?
addAddV2*batch_normalization_5/FusedBatchNormV3:y:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????-P 2
addW
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????-P 2
Relu?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:& *
dtype02?
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.conv2d_na_2/conv2d_2/kernel/Regularizer/SquareSquareEconv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:& 20
.conv2d_na_2/conv2d_2/kernel/Regularizer/Square?
-conv2d_na_2/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/Const?
+conv2d_na_2/conv2d_2/kernel/Regularizer/SumSum2conv2d_na_2/conv2d_2/kernel/Regularizer/Square:y:06conv2d_na_2/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/Sum?
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x?
+conv2d_na_2/conv2d_2/kernel/Regularizer/mulMul6conv2d_na_2/conv2d_2/kernel/Regularizer/mul/x:output:04conv2d_na_2/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+conv2d_na_2/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^add/ReadVariableOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp>^conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????-P 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????-P&::::::2(
add/ReadVariableOpadd/ReadVariableOp2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2~
=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=conv2d_na_2/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????-P&
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1439

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????F
conv2d_6:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?/
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?)
_tf_keras_network?){"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 180, 320, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2DFixed", "config": {"layer was saved without config": true}, "name": "conv2d_fixed", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate", "inbound_nodes": [[["conv2d_fixed", 0, 0, {}], ["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D_NA", "config": {"layer was saved without config": true}, "name": "conv2d_na", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Conv2DFixed", "config": {"layer was saved without config": true}, "name": "conv2d_fixed_1", "inbound_nodes": [[["conv2d_na", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.image.resize", "trainable": true, "dtype": "float32", "function": "image.resize"}, "name": "tf.image.resize", "inbound_nodes": [["concatenate", 0, 0, {"size": [90, 160], "antialias": false}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate_1", "inbound_nodes": [[["conv2d_fixed_1", 0, 0, {}], ["tf.image.resize", 0, 0, {}]]]}, {"class_name": "Conv2D_NA", "config": {"layer was saved without config": true}, "name": "conv2d_na_1", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Conv2DFixed", "config": {"layer was saved without config": true}, "name": "conv2d_fixed_2", "inbound_nodes": [[["conv2d_na_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.image.resize_1", "trainable": true, "dtype": "float32", "function": "image.resize"}, "name": "tf.image.resize_1", "inbound_nodes": [["concatenate", 0, 0, {"size": [45, 80], "antialias": false}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate_2", "inbound_nodes": [[["conv2d_fixed_2", 0, 0, {}], ["tf.image.resize_1", 0, 0, {}]]]}, {"class_name": "Conv2D_NA", "config": {"layer was saved without config": true}, "name": "conv2d_na_2", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Conv2D_NA", "config": {"layer was saved without config": true}, "name": "conv2d_na_3", "inbound_nodes": [[["conv2d_na_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate_3", "inbound_nodes": [[["conv2d_na_2", 0, 0, {}], ["conv2d_na_3", 0, 0, {}]]]}, {"class_name": "Conv2D_NA", "config": {"layer was saved without config": true}, "name": "conv2d_na_4", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Conv2DFixed_Transpose", "config": {"layer was saved without config": true}, "name": "conv2d_fixed__transpose", "inbound_nodes": [[["conv2d_na_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate_4", "inbound_nodes": [[["conv2d_na_1", 0, 0, {}], ["conv2d_fixed__transpose", 0, 0, {}]]]}, {"class_name": "Conv2D_NA", "config": {"layer was saved without config": true}, "name": "conv2d_na_5", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Conv2DFixed_Transpose", "config": {"layer was saved without config": true}, "name": "conv2d_fixed__transpose_1", "inbound_nodes": [[["conv2d_na_5", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["conv2d_na", 0, 0, {}], ["conv2d_fixed__transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 180, 320, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 180, 320, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 180, 320, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
strides
w
bn
regularization_losses
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DFixed", "name": "conv2d_fixed", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 3}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 180, 320, 3]}, {"class_name": "TensorShape", "items": [null, 180, 320, 3]}]}
?
&conv
'bn
(regularization_losses
)trainable_variables
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D_NA", "name": "conv2d_na", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
,strides
-pad
.w
/bn
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DFixed", "name": "conv2d_fixed_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
4	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.image.resize", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.image.resize", "trainable": true, "dtype": "float32", "function": "image.resize"}}
?
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": 3}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 90, 160, 8]}, {"class_name": "TensorShape", "items": [null, 90, 160, 6]}]}
?
9conv
:bn
;regularization_losses
<trainable_variables
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D_NA", "name": "conv2d_na_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
?strides
@pad
Aw
Bbn
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DFixed", "name": "conv2d_fixed_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
G	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.image.resize_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.image.resize_1", "trainable": true, "dtype": "float32", "function": "image.resize"}}
?
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": 3}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 45, 80, 32]}, {"class_name": "TensorShape", "items": [null, 45, 80, 6]}]}
?
Lconv
Mbn
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D_NA", "name": "conv2d_na_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
Rconv
Sbn
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D_NA", "name": "conv2d_na_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": 3}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 45, 80, 32]}, {"class_name": "TensorShape", "items": [null, 45, 80, 64]}]}
?
\conv
]bn
^regularization_losses
_trainable_variables
`	variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D_NA", "name": "conv2d_na_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
bstrides
cpad
d	out_shape
ew
fbn
gregularization_losses
htrainable_variables
i	variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DFixed_Transpose", "name": "conv2d_fixed__transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": 3}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 90, 160, 32]}, {"class_name": "TensorShape", "items": [null, 90, 160, 32]}]}
?
oconv
pbn
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D_NA", "name": "conv2d_na_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
ustrides
vpad
w	out_shape
xw
ybn
zregularization_losses
{trainable_variables
|	variables
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DFixed_Transpose", "name": "conv2d_fixed__transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
~regularization_losses
trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 180, 320, 8]}, {"class_name": "TensorShape", "items": [null, 180, 320, 32]}]}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 40]}}
(
?0"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35"
trackable_list_wrapper
?
?0
?1
2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
.13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
A24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
e47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
x58
?59
?60
?61
?62"
trackable_list_wrapper
?
regularization_losses
?layers
 ?layer_regularization_losses
trainable_variables
?layer_metrics
?metrics
	variables
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 :2Variable
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 3]}}
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
G
?0
?1
2
?3
?4"
trackable_list_wrapper
?
regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?metrics
 	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
#trainable_variables
?metrics
$	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 6]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 8]}}
(
?0"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
(regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
)trainable_variables
?metrics
*	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 :2Variable
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 8]}}
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
G
?0
?1
.2
?3
?4"
trackable_list_wrapper
?
0regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
1trainable_variables
?metrics
2	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
6trainable_variables
?metrics
7	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 14}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 14]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 32]}}
(
?0"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
;regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
<trainable_variables
?metrics
=	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 : 2Variable
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 80, 32]}}
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
G
?0
?1
A2
?3
?4"
trackable_list_wrapper
?
Cregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Dtrainable_variables
?metrics
E	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Itrainable_variables
?metrics
J	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 80, 38]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 80, 32]}}
(
?0"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
Nregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Otrainable_variables
?metrics
P	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 80, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 80, 64]}}
(
?0"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
Tregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Utrainable_variables
?metrics
V	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Ytrainable_variables
?metrics
Z	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 80, 96]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 80, 32]}}
(
?0"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
^regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
_trainable_variables
?metrics
`	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 : 2Variable
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 32]}}
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
G
?0
?1
e2
?3
?4"
trackable_list_wrapper
?
gregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
htrainable_variables
?metrics
i	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
kregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
ltrainable_variables
?metrics
m	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 160, 32]}}
(
?0"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
qregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
rtrainable_variables
?metrics
s	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 : 2Variable
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180, 320, 32]}}
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
G
?0
?1
x2
?3
?4"
trackable_list_wrapper
?
zregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
{trainable_variables
?metrics
|	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
~regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'(2conv2d_6/kernel
:2conv2d_6/bias
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:22&conv2d_fixed/batch_normalization/gamma
3:12%conv2d_fixed/batch_normalization/beta
1:/2conv2d_na/conv2d/kernel
#:!2conv2d_na/conv2d/bias
3:12%conv2d_na/batch_normalization_1/gamma
2:02$conv2d_na/batch_normalization_1/beta
8:62*conv2d_fixed_1/batch_normalization_2/gamma
7:52)conv2d_fixed_1/batch_normalization_2/beta
5:3 2conv2d_na_1/conv2d_1/kernel
':% 2conv2d_na_1/conv2d_1/bias
5:3 2'conv2d_na_1/batch_normalization_3/gamma
4:2 2&conv2d_na_1/batch_normalization_3/beta
8:6 2*conv2d_fixed_2/batch_normalization_4/gamma
7:5 2)conv2d_fixed_2/batch_normalization_4/beta
5:3& 2conv2d_na_2/conv2d_2/kernel
':% 2conv2d_na_2/conv2d_2/bias
5:3 2'conv2d_na_2/batch_normalization_5/gamma
4:2 2&conv2d_na_2/batch_normalization_5/beta
5:3 @2conv2d_na_3/conv2d_3/kernel
':%@2conv2d_na_3/conv2d_3/bias
5:3@2'conv2d_na_3/batch_normalization_6/gamma
4:2@2&conv2d_na_3/batch_normalization_6/beta
5:3` 2conv2d_na_4/conv2d_4/kernel
':% 2conv2d_na_4/conv2d_4/bias
5:3 2'conv2d_na_4/batch_normalization_7/gamma
4:2 2&conv2d_na_4/batch_normalization_7/beta
A:? 23conv2d_fixed__transpose/batch_normalization_8/gamma
@:> 22conv2d_fixed__transpose/batch_normalization_8/beta
5:3@ 2conv2d_na_5/conv2d_5/kernel
':% 2conv2d_na_5/conv2d_5/bias
5:3 2'conv2d_na_5/batch_normalization_9/gamma
4:2 2&conv2d_na_5/batch_normalization_9/beta
D:B 26conv2d_fixed__transpose_1/batch_normalization_10/gamma
C:A 25conv2d_fixed__transpose_1/batch_normalization_10/beta
<:: (2,conv2d_fixed/batch_normalization/moving_mean
@:> (20conv2d_fixed/batch_normalization/moving_variance
;:9 (2+conv2d_na/batch_normalization_1/moving_mean
?:= (2/conv2d_na/batch_normalization_1/moving_variance
@:> (20conv2d_fixed_1/batch_normalization_2/moving_mean
D:B (24conv2d_fixed_1/batch_normalization_2/moving_variance
=:;  (2-conv2d_na_1/batch_normalization_3/moving_mean
A:?  (21conv2d_na_1/batch_normalization_3/moving_variance
@:>  (20conv2d_fixed_2/batch_normalization_4/moving_mean
D:B  (24conv2d_fixed_2/batch_normalization_4/moving_variance
=:;  (2-conv2d_na_2/batch_normalization_5/moving_mean
A:?  (21conv2d_na_2/batch_normalization_5/moving_variance
=:;@ (2-conv2d_na_3/batch_normalization_6/moving_mean
A:?@ (21conv2d_na_3/batch_normalization_6/moving_variance
=:;  (2-conv2d_na_4/batch_normalization_7/moving_mean
A:?  (21conv2d_na_4/batch_normalization_7/moving_variance
I:G  (29conv2d_fixed__transpose/batch_normalization_8/moving_mean
M:K  (2=conv2d_fixed__transpose/batch_normalization_8/moving_variance
=:;  (2-conv2d_na_5/batch_normalization_9/moving_mean
A:?  (21conv2d_na_5/batch_normalization_9/moving_variance
L:J  (2<conv2d_fixed__transpose_1/batch_normalization_10/moving_mean
P:N  (2@conv2d_fixed__transpose_1/batch_normalization_10/moving_variance
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
?1
?2
?3
?4
.5
?6
?7
?8
?9
A10
?11
?12
?13
?14
?15
?16
?17
?18
e19
?20
?21
?22
?23
x24
?25
?26"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
7
0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
7
.0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
7
A0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
f0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
7
e0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?metrics
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
7
x0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?2?
$__inference_model_layer_call_fn_5278
$__inference_model_layer_call_fn_5409
$__inference_model_layer_call_fn_4025
$__inference_model_layer_call_fn_4350?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_model_layer_call_and_return_conditional_losses_3699
?__inference_model_layer_call_and_return_conditional_losses_5147
?__inference_model_layer_call_and_return_conditional_losses_4847
?__inference_model_layer_call_and_return_conditional_losses_3505?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_1034?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_1???????????
?2?
+__inference_conv2d_fixed_layer_call_fn_5483
+__inference_conv2d_fixed_layer_call_fn_5468?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_5432
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_5453?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_concatenate_layer_call_fn_5496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_concatenate_layer_call_and_return_conditional_losses_5490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_na_layer_call_fn_5604
(__inference_conv2d_na_layer_call_fn_5587?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_na_layer_call_and_return_conditional_losses_5537
C__inference_conv2d_na_layer_call_and_return_conditional_losses_5570?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv2d_fixed_1_layer_call_fn_5678
-__inference_conv2d_fixed_1_layer_call_fn_5663?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_5648
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_5627?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_1_layer_call_fn_5691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_1_layer_call_and_return_conditional_losses_5685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_na_1_layer_call_fn_5799
*__inference_conv2d_na_1_layer_call_fn_5782?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_5732
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_5765?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv2d_fixed_2_layer_call_fn_5858
-__inference_conv2d_fixed_2_layer_call_fn_5873?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_5822
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_5843?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_2_layer_call_fn_5886?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_2_layer_call_and_return_conditional_losses_5880?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_na_2_layer_call_fn_5977
*__inference_conv2d_na_2_layer_call_fn_5994?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_5960
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_5927?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_na_3_layer_call_fn_6085
*__inference_conv2d_na_3_layer_call_fn_6102?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_6068
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_6035?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_3_layer_call_fn_6115?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_3_layer_call_and_return_conditional_losses_6109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_na_4_layer_call_fn_6206
*__inference_conv2d_na_4_layer_call_fn_6223?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_6156
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_6189?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_conv2d_fixed__transpose_layer_call_fn_6284
6__inference_conv2d_fixed__transpose_layer_call_fn_6299?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_6269
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_6247?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_4_layer_call_fn_6312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_4_layer_call_and_return_conditional_losses_6306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_na_5_layer_call_fn_6420
*__inference_conv2d_na_5_layer_call_fn_6403?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_6386
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_6353?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_conv2d_fixed__transpose_1_layer_call_fn_6496
8__inference_conv2d_fixed__transpose_1_layer_call_fn_6481?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_6444
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_6466?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_5_layer_call_fn_6509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_5_layer_call_and_return_conditional_losses_6503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_6_layer_call_fn_6547?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_6538?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_6558?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
"__inference_signature_wrapper_4525input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_batch_normalization_layer_call_fn_6622
2__inference_batch_normalization_layer_call_fn_6609?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6596
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6578?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_1_layer_call_fn_6692
4__inference_batch_normalization_1_layer_call_fn_6679?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6648
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6666?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_1_6703?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
4__inference_batch_normalization_2_layer_call_fn_6767
4__inference_batch_normalization_2_layer_call_fn_6754?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6723
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6741?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_3_layer_call_fn_6824
4__inference_batch_normalization_3_layer_call_fn_6837?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6793
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6811?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_2_6848?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
4__inference_batch_normalization_4_layer_call_fn_6899
4__inference_batch_normalization_4_layer_call_fn_6912?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6886
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6868?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_5_layer_call_fn_6982
4__inference_batch_normalization_5_layer_call_fn_6969?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6956
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6938?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_3_6993?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_6_layer_call_fn_7063
4__inference_batch_normalization_6_layer_call_fn_7050?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7019
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7037?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_4_7074?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_7_layer_call_fn_7131
4__inference_batch_normalization_7_layer_call_fn_7144?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7100
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7118?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_5_7155?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
4__inference_batch_normalization_8_layer_call_fn_7219
4__inference_batch_normalization_8_layer_call_fn_7206?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7193
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7175?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_9_layer_call_fn_7289
4__inference_batch_normalization_9_layer_call_fn_7276?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_7245
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_7263?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_6_7300?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
5__inference_batch_normalization_10_layer_call_fn_7364
5__inference_batch_normalization_10_layer_call_fn_7351?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7338
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7320?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
__inference__wrapped_model_1034?y??????????.??????????A??????????????????????e??????????x??????:?7
0?-
+?(
input_1???????????
? "=?:
8
conv2d_6,?)
conv2d_6????????????
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7320?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7338?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_batch_normalization_10_layer_call_fn_7351?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
5__inference_batch_normalization_10_layer_call_fn_7364?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6648?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6666?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
4__inference_batch_normalization_1_layer_call_fn_6679?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
4__inference_batch_normalization_1_layer_call_fn_6692?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6723?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6741?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
4__inference_batch_normalization_2_layer_call_fn_6754?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
4__inference_batch_normalization_2_layer_call_fn_6767?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6793?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6811?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_3_layer_call_fn_6824?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_3_layer_call_fn_6837?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6868?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6886?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_4_layer_call_fn_6899?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_4_layer_call_fn_6912?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6938?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6956?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_5_layer_call_fn_6969?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_5_layer_call_fn_6982?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7019?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7037?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
4__inference_batch_normalization_6_layer_call_fn_7050?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
4__inference_batch_normalization_6_layer_call_fn_7063?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7100?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7118?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_7_layer_call_fn_7131?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_7_layer_call_fn_7144?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7175?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7193?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_8_layer_call_fn_7206?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_8_layer_call_fn_7219?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_7245?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_7263?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_9_layer_call_fn_7276?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_9_layer_call_fn_7289?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6578?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6596?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
2__inference_batch_normalization_layer_call_fn_6609?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
2__inference_batch_normalization_layer_call_fn_6622?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
G__inference_concatenate_1_layer_call_and_return_conditional_losses_5685?l?i
b?_
]?Z
+?(
inputs/0?????????Z?
+?(
inputs/1?????????Z?
? ".?+
$?!
0?????????Z?
? ?
,__inference_concatenate_1_layer_call_fn_5691?l?i
b?_
]?Z
+?(
inputs/0?????????Z?
+?(
inputs/1?????????Z?
? "!??????????Z??
G__inference_concatenate_2_layer_call_and_return_conditional_losses_5880?j?g
`?]
[?X
*?'
inputs/0?????????-P 
*?'
inputs/1?????????-P
? "-?*
#? 
0?????????-P&
? ?
,__inference_concatenate_2_layer_call_fn_5886?j?g
`?]
[?X
*?'
inputs/0?????????-P 
*?'
inputs/1?????????-P
? " ??????????-P&?
G__inference_concatenate_3_layer_call_and_return_conditional_losses_6109?j?g
`?]
[?X
*?'
inputs/0?????????-P 
*?'
inputs/1?????????-P@
? "-?*
#? 
0?????????-P`
? ?
,__inference_concatenate_3_layer_call_fn_6115?j?g
`?]
[?X
*?'
inputs/0?????????-P 
*?'
inputs/1?????????-P@
? " ??????????-P`?
G__inference_concatenate_4_layer_call_and_return_conditional_losses_6306?l?i
b?_
]?Z
+?(
inputs/0?????????Z? 
+?(
inputs/1?????????Z? 
? ".?+
$?!
0?????????Z?@
? ?
,__inference_concatenate_4_layer_call_fn_6312?l?i
b?_
]?Z
+?(
inputs/0?????????Z? 
+?(
inputs/1?????????Z? 
? "!??????????Z?@?
G__inference_concatenate_5_layer_call_and_return_conditional_losses_6503?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1??????????? 
? "/?,
%?"
0???????????(
? ?
,__inference_concatenate_5_layer_call_fn_6509?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1??????????? 
? ""????????????(?
E__inference_concatenate_layer_call_and_return_conditional_losses_5490?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? "/?,
%?"
0???????????
? ?
*__inference_concatenate_layer_call_fn_5496?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""?????????????
B__inference_conv2d_6_layer_call_and_return_conditional_losses_6538r??9?6
/?,
*?'
inputs???????????(
? "/?,
%?"
0???????????
? ?
'__inference_conv2d_6_layer_call_fn_6547e??9?6
/?,
*?'
inputs???????????(
? ""?????????????
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_5627z	.????=?:
3?0
*?'
inputs???????????
p
? ".?+
$?!
0?????????Z?
? ?
H__inference_conv2d_fixed_1_layer_call_and_return_conditional_losses_5648z	.????=?:
3?0
*?'
inputs???????????
p 
? ".?+
$?!
0?????????Z?
? ?
-__inference_conv2d_fixed_1_layer_call_fn_5663m	.????=?:
3?0
*?'
inputs???????????
p
? "!??????????Z??
-__inference_conv2d_fixed_1_layer_call_fn_5678m	.????=?:
3?0
*?'
inputs???????????
p 
? "!??????????Z??
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_5822x	A????<?9
2?/
)?&
inputs?????????Z? 
p
? "-?*
#? 
0?????????-P 
? ?
H__inference_conv2d_fixed_2_layer_call_and_return_conditional_losses_5843x	A????<?9
2?/
)?&
inputs?????????Z? 
p 
? "-?*
#? 
0?????????-P 
? ?
-__inference_conv2d_fixed_2_layer_call_fn_5858k	A????<?9
2?/
)?&
inputs?????????Z? 
p
? " ??????????-P ?
-__inference_conv2d_fixed_2_layer_call_fn_5873k	A????<?9
2?/
)?&
inputs?????????Z? 
p 
? " ??????????-P ?
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_6444z	x????<?9
2?/
)?&
inputs?????????Z? 
p
? "/?,
%?"
0??????????? 
? ?
S__inference_conv2d_fixed__transpose_1_layer_call_and_return_conditional_losses_6466z	x????<?9
2?/
)?&
inputs?????????Z? 
p 
? "/?,
%?"
0??????????? 
? ?
8__inference_conv2d_fixed__transpose_1_layer_call_fn_6481m	x????<?9
2?/
)?&
inputs?????????Z? 
p
? ""???????????? ?
8__inference_conv2d_fixed__transpose_1_layer_call_fn_6496m	x????<?9
2?/
)?&
inputs?????????Z? 
p 
? ""???????????? ?
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_6247x	e????;?8
1?.
(?%
inputs?????????-P 
p
? ".?+
$?!
0?????????Z? 
? ?
Q__inference_conv2d_fixed__transpose_layer_call_and_return_conditional_losses_6269x	e????;?8
1?.
(?%
inputs?????????-P 
p 
? ".?+
$?!
0?????????Z? 
? ?
6__inference_conv2d_fixed__transpose_layer_call_fn_6284k	e????;?8
1?.
(?%
inputs?????????-P 
p
? "!??????????Z? ?
6__inference_conv2d_fixed__transpose_layer_call_fn_6299k	e????;?8
1?.
(?%
inputs?????????-P 
p 
? "!??????????Z? ?
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_5432{	????=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
F__inference_conv2d_fixed_layer_call_and_return_conditional_losses_5453{	????=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_fixed_layer_call_fn_5468n	????=?:
3?0
*?'
inputs???????????
p
? ""?????????????
+__inference_conv2d_fixed_layer_call_fn_5483n	????=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_5732|??????<?9
2?/
)?&
inputs?????????Z?
p
? ".?+
$?!
0?????????Z? 
? ?
E__inference_conv2d_na_1_layer_call_and_return_conditional_losses_5765|??????<?9
2?/
)?&
inputs?????????Z?
p 
? ".?+
$?!
0?????????Z? 
? ?
*__inference_conv2d_na_1_layer_call_fn_5782o??????<?9
2?/
)?&
inputs?????????Z?
p
? "!??????????Z? ?
*__inference_conv2d_na_1_layer_call_fn_5799o??????<?9
2?/
)?&
inputs?????????Z?
p 
? "!??????????Z? ?
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_5927z??????;?8
1?.
(?%
inputs?????????-P&
p
? "-?*
#? 
0?????????-P 
? ?
E__inference_conv2d_na_2_layer_call_and_return_conditional_losses_5960z??????;?8
1?.
(?%
inputs?????????-P&
p 
? "-?*
#? 
0?????????-P 
? ?
*__inference_conv2d_na_2_layer_call_fn_5977m??????;?8
1?.
(?%
inputs?????????-P&
p
? " ??????????-P ?
*__inference_conv2d_na_2_layer_call_fn_5994m??????;?8
1?.
(?%
inputs?????????-P&
p 
? " ??????????-P ?
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_6035z??????;?8
1?.
(?%
inputs?????????-P 
p
? "-?*
#? 
0?????????-P@
? ?
E__inference_conv2d_na_3_layer_call_and_return_conditional_losses_6068z??????;?8
1?.
(?%
inputs?????????-P 
p 
? "-?*
#? 
0?????????-P@
? ?
*__inference_conv2d_na_3_layer_call_fn_6085m??????;?8
1?.
(?%
inputs?????????-P 
p
? " ??????????-P@?
*__inference_conv2d_na_3_layer_call_fn_6102m??????;?8
1?.
(?%
inputs?????????-P 
p 
? " ??????????-P@?
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_6156z??????;?8
1?.
(?%
inputs?????????-P`
p
? "-?*
#? 
0?????????-P 
? ?
E__inference_conv2d_na_4_layer_call_and_return_conditional_losses_6189z??????;?8
1?.
(?%
inputs?????????-P`
p 
? "-?*
#? 
0?????????-P 
? ?
*__inference_conv2d_na_4_layer_call_fn_6206m??????;?8
1?.
(?%
inputs?????????-P`
p
? " ??????????-P ?
*__inference_conv2d_na_4_layer_call_fn_6223m??????;?8
1?.
(?%
inputs?????????-P`
p 
? " ??????????-P ?
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_6353|??????<?9
2?/
)?&
inputs?????????Z?@
p
? ".?+
$?!
0?????????Z? 
? ?
E__inference_conv2d_na_5_layer_call_and_return_conditional_losses_6386|??????<?9
2?/
)?&
inputs?????????Z?@
p 
? ".?+
$?!
0?????????Z? 
? ?
*__inference_conv2d_na_5_layer_call_fn_6403o??????<?9
2?/
)?&
inputs?????????Z?@
p
? "!??????????Z? ?
*__inference_conv2d_na_5_layer_call_fn_6420o??????<?9
2?/
)?&
inputs?????????Z?@
p 
? "!??????????Z? ?
C__inference_conv2d_na_layer_call_and_return_conditional_losses_5537~??????=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
C__inference_conv2d_na_layer_call_and_return_conditional_losses_5570~??????=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
(__inference_conv2d_na_layer_call_fn_5587q??????=?:
3?0
*?'
inputs???????????
p
? ""?????????????
(__inference_conv2d_na_layer_call_fn_5604q??????=?:
3?0
*?'
inputs???????????
p 
? ""????????????:
__inference_loss_fn_0_6558??

? 
? "? :
__inference_loss_fn_1_6703??

? 
? "? :
__inference_loss_fn_2_6848??

? 
? "? :
__inference_loss_fn_3_6993??

? 
? "? :
__inference_loss_fn_4_7074??

? 
? "? :
__inference_loss_fn_5_7155??

? 
? "? :
__inference_loss_fn_6_7300??

? 
? "? ?
?__inference_model_layer_call_and_return_conditional_losses_3505?y??????????.??????????A??????????????????????e??????????x??????B??
8?5
+?(
input_1???????????
p

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3699?y??????????.??????????A??????????????????????e??????????x??????B??
8?5
+?(
input_1???????????
p 

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_4847?y??????????.??????????A??????????????????????e??????????x??????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_5147?y??????????.??????????A??????????????????????e??????????x??????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
$__inference_model_layer_call_fn_4025?y??????????.??????????A??????????????????????e??????????x??????B??
8?5
+?(
input_1???????????
p

 
? ""?????????????
$__inference_model_layer_call_fn_4350?y??????????.??????????A??????????????????????e??????????x??????B??
8?5
+?(
input_1???????????
p 

 
? ""?????????????
$__inference_model_layer_call_fn_5278?y??????????.??????????A??????????????????????e??????????x??????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
$__inference_model_layer_call_fn_5409?y??????????.??????????A??????????????????????e??????????x??????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
"__inference_signature_wrapper_4525?y??????????.??????????A??????????????????????e??????????x??????E?B
? 
;?8
6
input_1+?(
input_1???????????"=?:
8
conv2d_6,?)
conv2d_6???????????