#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP 

namespace dpp {

class Sequential {
	public:
		Sequential() { std::cout << std::fixed << std::setprecision(8); }

		/* Destructor : If there are layers, reset their pointer and clear the vector */
		~Sequential() { 
			if (size_ > 0) { 
				std::for_each(network_.begin(), network_.end(), [](auto &layer) { layer.reset(); });
				network_.clear();
			} 
		}

	protected:

};
}
