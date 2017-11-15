#include <iostream>
#include <string>
#include <ctime>

#include <complex>

//#include <mkl.h>

#include <armadillo>

//typedef std::complex<double> element_type;
typedef double element_type;

int main(int argc, char*argv[])
{
	if (argc <= 1)
	{
		std::cout << "Matrix size is not given" << std::endl;
		return 0;
	}
	
	//std::cout << "MKL uses " << mkl_get_max_threads() << " threads" << std::endl;
	
	clock_t my_time;
	const int matrix_size = std::stoi(argv[1]);

	//randomly assign a matrix
	my_time = clock();

	arma::Mat<element_type> A = arma::randu<arma::Mat<element_type>>(matrix_size, matrix_size);

	std::cout << "Randomly assigning a matrix of size "
		<< matrix_size << " costs " << 1000 * ((clock() - (float)my_time) / CLOCKS_PER_SEC) << " ms" << std::endl;


	//matrix multiplication
	my_time = clock();

	arma::Mat<element_type> B = A.t()*A;

	std::cout << "Multiplying two matrices of size "
		<< matrix_size << " costs " << 1000 * ((clock() - (float)my_time) / CLOCKS_PER_SEC) << " ms" << std::endl;


	//diagonalization
	//arma::Col<element_type> eigval(matrix_size);
	arma::Col<double> eigval(matrix_size);
	arma::Mat<element_type> eigvec(matrix_size, matrix_size);

	my_time = clock();

	arma::eig_sym(eigval, eigvec, B);

	std::cout << "Diagonalizing a matrix of size "
		<< matrix_size << " costs " << 1000 * ((clock() - (float)my_time) / CLOCKS_PER_SEC) << " ms" << std::endl;


	return 0;
}
