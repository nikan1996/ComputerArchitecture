(base) nikan@nikan-desktop:~/assignments/assignment2-2018$ nosetests -v tests/test_tvm_op.py
test_tvm_op.test_matrix_elementwise_add ... ok
test_tvm_op.test_matrix_elementwise_add_by_const ... ok
test_tvm_op.test_matrix_elementwise_mul ... ok
test_tvm_op.test_matrix_elementwise_mul_by_const ... ok
test_tvm_op.test_conv2d ... ok
test_tvm_op.test_relu ... ok
test_tvm_op.test_relu_gradient ... ok
test_tvm_op.test_softmax ... ok
test_tvm_op.test_softmax_cross_entropy ... ok
test_tvm_op.test_reduce_sum_axis_zero ... ok
test_tvm_op.test_broadcast_to ... ok

----------------------------------------------------------------------
Ran 11 tests in 2.992s


### Grading rubrics
- test_tvm_op.test_matrix_elementwise_add ... Implemented by us, not graded.
- test_tvm_op.test_matrix_elementwise_add_by_const ... 1pt
- test_tvm_op.test_matrix_elementwise_mul ... 1pt
- test_tvm_op.test_matrix_elementwise_mul_by_const ... 1pt
- test_tvm_op.test_matrix_multiply ... 2pt
- test_tvm_op.test_conv2d ... 2pt
- test_tvm_op.test_relu ... 1pt
- test_tvm_op.test_relu_gradient ... 1pt
- test_tvm_op.test_softmax ... 1pt
- test_tvm_op.test_softmax_cross_entropy ... 2pt
- test_tvm_op.test_reduce_sum_axis_zero ... Implemented by us, not graded.
- test_tvm_op.test_broadcast_to ... Implemented by us, not graded.
实现了12pt的部分。