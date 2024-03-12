# Use an official TensorFlow Serving runtime as a parent image
FROM tensorflow/serving:latest

# Copy the models and configuration file into the container
COPY 1 /models/vectrino/1
COPY 2 /models/vectrino/2
COPY tf.config /models/vectrino/tf.config

# Expose the gRPC and HTTP ports
EXPOSE 8500
EXPOSE 8501

# Start TensorFlow Serving
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
CMD ["--model_name=vectrino", "--port=8500", "--rest_api_port=8501", "--model_config_file=/models/vectrino/tf.config"]
