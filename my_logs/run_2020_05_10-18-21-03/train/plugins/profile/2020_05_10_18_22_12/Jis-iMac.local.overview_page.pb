�	ˡE����?ˡE����?!ˡE����?	 ��184@ ��184@! ��184@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ˡE����?�� �rh�?A�v��/�?YL7�A`��?*	     @k@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate��(\�µ?!�'K`�~C@)ˡE����?1�Ṱ�B@:Preprocessing2F
Iterator::Model��(\�µ?!�'K`�~C@)�l����?1���A��@@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatX9��v��?!��p,@)y�&1��?1���%�)@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�/�$�?!ش�,�N@)���Q��?1^8�߅@:Preprocessing2S
Iterator::Model::ParallelMap�I+��?!��A��.@)�I+��?1��A��.@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�h?!L`�~��?)�~j�t�h?1L`�~��?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!L`�~��?)�~j�t�h?1L`�~��?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��ʡE�?!D���A�C@)����Mb`?1����[�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 20.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2B10.4 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�� �rh�?�� �rh�?!�� �rh�?      ��!       "      ��!       *      ��!       2	�v��/�?�v��/�?!�v��/�?:      ��!       B      ��!       J	L7�A`��?L7�A`��?!L7�A`��?R      ��!       Z	L7�A`��?L7�A`��?!L7�A`��?JCPU_ONLY2black"�
host�Your program is HIGHLY input-bound because 20.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendationR
nomoderate"B10.4 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 