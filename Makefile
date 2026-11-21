CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++20 -Wall -Wextra -Wno-uninitialized
THREADS = -pthread
HEADERS = integer.hpp parallel.hpp ntt.hpp wide.hpp short_fft.hpp leaf.hpp factored.hpp
SOURCES = pi.cpp integer.cpp parallel.cpp

# Train and use the profile in the same directory so GCC can find every counter.
pi_cpu: Makefile $(SOURCES) $(HEADERS)
	@set -eu; work=$$(mktemp -d); trap 'rm -rf "$$work"' EXIT; \
	cp $^ "$$work/"; \
	$(MAKE) -C "$$work" cpu-build CXXFLAGS="$(CXXFLAGS) -fprofile-generate=$$work/profile -fprofile-update=atomic"; \
	"$$work/cpu-build" 60000000 --benchmark --threads 8 >/dev/null; \
	$(MAKE) -B -C "$$work" cpu-build CXXFLAGS="$(CXXFLAGS) -fprofile-use=$$work/profile -fprofile-correction -fprofile-partial-training"; \
	cp "$$work/cpu-build" "$@"

cpu-build: pi.cpp $(HEADERS) integer.o parallel.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREADS) pi.cpp integer.o parallel.o $(LDFLAGS) -lcrypto -o $@

integer.o: integer.cpp integer.hpp parallel.hpp ntt.hpp wide.hpp short_fft.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREADS) -c $< -o $@

parallel.o: parallel.cpp parallel.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREADS) -c $< -o $@

check-integer: check.cpp integer.o parallel.o integer.hpp parallel.hpp leaf.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREADS) check.cpp integer.o parallel.o $(LDFLAGS) -o $@

.PHONY: check clean
check: pi_cpu check-integer
	./check-integer
	PI_TRANSFORM_MIB=1 ./check-integer
	python3 check_pi.py

clean:
	$(RM) pi_cpu cpu-build check-integer integer.o parallel.o

.DELETE_ON_ERROR:
