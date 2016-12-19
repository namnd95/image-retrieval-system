import uuid
import ir.settings
import os

def save_image(image):
    filename = str(uuid.uuid4())
    destination = open(os.path.join(ir.settings.MEDIA_ROOT,'query', filename), 'wb+')
    for chunk in image.chunks():
        destination.write(chunk)
    destination.close()
    return 'query/' + filename

def run(image):
    name = save_image(image)

    return {
        'query' : name,
        'result' : ['radcliffe_camera_1.jpg', 'radcliffe_camera_2.jpg']
    }
