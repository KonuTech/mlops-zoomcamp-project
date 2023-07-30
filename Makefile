setup:
	make java-install && \
	make spark-install && \
	make pyspark-install && \
	make install-dependencies

java-install:
	mkdir -p spark
	wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz -P ~/spark
	tar xzfv ~/spark/openjdk-11.0.2_linux-x64_bin.tar.gz -C ~/spark/
	echo 'export JAVA_HOME="${HOME}/spark/jdk-11.0.2"' >> ~/.bashrc
	echo 'export PATH="${JAVA_HOME}/bin:${PATH}"' >> ~/.bashrc
	rm ~/spark/openjdk-11.0.2_linux-x64_bin.tar.gz

spark-install:
	wget https://dlcdn.apache.org/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz -P ~/spark
	tar xzfv ~/spark/spark-3.3.2-bin-hadoop3.tgz -C ~/spark/
	rm ~/spark/spark-3.3.2-bin-hadoop3.tgz
	echo 'export SPARK_HOME="${HOME}/spark/spark-3.3.2-bin-hadoop3"' >> ~/.bashrc
	echo 'export PATH="${SPARK_HOME}/bin:${PATH}"' >> ~/.bashrc

pyspark-install:
	echo 'export PYTHONPATH="${SPARK_HOME}/python/:$PYTHONPATH"' >> ~/.bashrc
	echo 'export PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.9.5-src.zip:$PYTHONPATH"' >> ~/.bashrc

install-dependencies:
	pip install --upgrade pip
	pip install -r requirements.txt
