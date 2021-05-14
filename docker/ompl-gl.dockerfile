FROM ompl_bionic as BUILDER

FROM nvidia/cudagl:11.1.1-runtime-ubuntu18.04 AS BASE

COPY --from=BUILDER /usr/local/include/ompl /usr/local/include/ompl
COPY --from=BUILDER /usr/local/lib/libompl* /usr/local/lib/
COPY --from=BUILDER /usr/local/share/ompl /usr/local/share/ompl
COPY --from=BUILDER /usr/local/bin/ompl_benchmark_statistics.py /usr/local/bin/ompl_benchmark_statistics.py
COPY --from=BUILDER /usr/local/share/man/man1/ompl_benchmark_statistics.1 /usr/local/share/man/man1/ompl_benchmark_statistics.1
COPY --from=BUILDER /usr/local/share/man/man1/plannerarena.1 /usr/local/share/man/man1/plannerarena.1
COPY --from=BUILDER /root/ompl /root/ompl
COPY --from=BUILDER /usr/local/lib/pkgconfig/ompl.pc /usr/local/lib/pkgconfig/ompl.pc
COPY --from=BUILDER /usr/lib/python3/dist-packages/ompl /usr/lib/python3/dist-packages/ompl

# Files required for OMPL
RUN apt-get update && apt-get install -y \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-numpy-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-python-dev \
    libboost-test-dev \
    libflann-dev \
    libode-dev \
    libeigen3-dev \
	python3-pip\
	&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
	git \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /root/.
WORKDIR /root
RUN pip3 install -r requirements.txt
RUN pip3 install GPy==1.9.9

RUN apt-get update && apt-get install -y \
	python3-pyqt5\
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/prob_planning

CMD [bash]