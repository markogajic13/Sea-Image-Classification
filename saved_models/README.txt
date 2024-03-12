http://localhost:8501/v1/models/vectrino/versions/1
docker run -d -p 8502:8501 mgajic/vectrino_tf:1.0.0
docker run -it -p 8501:8501 mgajic/tf_vectrino:1.0.0
