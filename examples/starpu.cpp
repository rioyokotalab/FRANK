#include "starpu.h"

#include <vector>

#include <iostream>


class Runtime {
 private:
  starpu_data_handle_t handle;
 public:
  Runtime() {
    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "init");
  }

  ~Runtime() {
    starpu_shutdown();
  }
};

static Runtime runtime;


class Matrix {
 private:
  starpu_data_handle_t handle_;
 public:
  Matrix() {
    starpu_matrix_data_register(&handle_, -1, 0, 4, 4, 4, sizeof(double));
    // starpu_vector_data_register(
    //   &handle_, STARPU_MAIN_RAM, (uintptr_t)&data[0], 16,  sizeof(double)
    // );
  }

  ~Matrix() {
    starpu_data_unregister_submit(handle_);
  }

  starpu_data_handle_t& handle() { return handle_; }
};

void init_cpu(void* buffers[], void*) {
	double* val = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
	uint nx = STARPU_MATRIX_GET_NX(buffers[0]);
  uint ny = STARPU_MATRIX_GET_NY(buffers[0]);
  uint ld = STARPU_MATRIX_GET_LD(buffers[0]);
  for (uint i=0; i<ny; ++i) {
    for (uint j=0; j<nx; ++j) {
      val[i*ld+j] = i+j;
    }
  }
}

void operation_cpu(void* buffers[], void* cl_args) {
	double* A_val = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
	uint A_nx = STARPU_MATRIX_GET_NX(buffers[0]);
  uint A_ny = STARPU_MATRIX_GET_NY(buffers[0]);
  uint A_ld = STARPU_MATRIX_GET_LD(buffers[0]);
	double* B_val = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
	uint B_nx = STARPU_MATRIX_GET_NX(buffers[1]);
  uint B_ny = STARPU_MATRIX_GET_NY(buffers[1]);
  uint B_ld = STARPU_MATRIX_GET_LD(buffers[1]);
  for (uint i=0; i<A_ny; ++i) {
    for (uint j=0; j<A_nx; ++j) {
      B_val[i*B_ld+j] += A_val[i*B_ld+j];
    }
  }
}

void print_cpu(void* buffers[], void*) {
	double* val = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
	uint nx = STARPU_MATRIX_GET_NX(buffers[0]);
  uint ny = STARPU_MATRIX_GET_NY(buffers[0]);
  uint ld = STARPU_MATRIX_GET_LD(buffers[0]);
  std::cout << std::endl;
  for (uint i=0; i<ny; ++i) {
    std::cout << "| " << val[i*ld];
    for (uint j=1; j<nx; ++j) {
       std::cout << ", " << val[i*ld+j];
    }
    std::cout << " |" << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  struct starpu_codelet init_cl;
  starpu_codelet_init(&init_cl);
  init_cl.cpu_funcs[0] = init_cpu;
  init_cl.cpu_funcs_name[0] = "init";
  init_cl.name = "init_cl";
  init_cl.nbuffers = 1;
  init_cl.modes[0] = STARPU_W;

  struct starpu_codelet operation_cl;
  starpu_codelet_init(&operation_cl);
  operation_cl.cpu_funcs[0] = operation_cpu;
  operation_cl.cpu_funcs_name[0] = "operation";
  operation_cl.name = "operation_cl";
  operation_cl.nbuffers = 2;
  operation_cl.modes[0] = STARPU_R;
  operation_cl.modes[1] = STARPU_RW;

  struct starpu_codelet print_cl;
  starpu_codelet_init(&print_cl);
  print_cl.cpu_funcs[0] = print_cpu;
  print_cl.cpu_funcs_name[0] = "print";
  print_cl.name = "print_cl";
  print_cl.nbuffers = 1;
  print_cl.modes[0] = STARPU_R;

  Matrix test1, test2, test3;


  struct starpu_task* init1 = starpu_task_create();
  init1->cl = &init_cl;
  init1->handles[0] = test1.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(init1), "init1");

  struct starpu_task* init2 = starpu_task_create();
  init2->cl = &init_cl;
  init2->handles[0] = test2.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(init2), "init2");

  struct starpu_task* init3 = starpu_task_create();
  init3->cl = &init_cl;
  init3->handles[0] = test3.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(init3), "init3");

  // starpu_task_wait_for_all();

  struct starpu_task* task1 = starpu_task_create();
  task1->cl = &operation_cl;
  task1->handles[0] = test1.handle();
  task1->handles[1] = test2.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task1), "task1");

  struct starpu_task* task2 = starpu_task_create();
  task2->cl = &operation_cl;
  task2->handles[0] = test1.handle();
  task2->handles[1] = test3.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task2), "task2");

  struct starpu_task* task3 = starpu_task_create();
  task3->cl = &operation_cl;
  task3->handles[0] = test2.handle();
  task3->handles[1] = test3.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task3), "task3");

  // starpu_task_wait_for_all();

  struct starpu_task* print1 = starpu_task_create();
  print1->cl = &print_cl;
  print1->handles[0] = test1.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(print1), "print1");
  starpu_task_wait_for_all();

  struct starpu_task* print2 = starpu_task_create();
  print2->cl = &print_cl;
  print2->handles[0] = test2.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(print2), "print2");
  starpu_task_wait_for_all();

  struct starpu_task* print3 = starpu_task_create();
  print3->cl = &print_cl;
  print3->handles[0] = test3.handle();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(print3), "print3");
  starpu_task_wait_for_all();

  return 0;
}
