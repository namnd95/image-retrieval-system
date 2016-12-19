import uuid

import ir.settings
import os
import get_image
import build_index
import get_image

def save_image(image):
    filename = str(uuid.uuid4())
    destination = open(os.path.join(ir.settings.MEDIA_ROOT,'query', filename), 'wb+')
    for chunk in image.chunks():
        destination.write(chunk)
    destination.close()
    return 'query/' + filename
    
def transform(id):
    name = build_index.data[id]
    name = name[name.find('media/')+6:]
    return name

def run(image):
    name = save_image(image)
    result_id = get_image.GetNeareastNeighbor().find(image, 10)
    result = []
    for i in result_id[0]:
        result.append(transform(i))

    return {
        'query' : name,
        'result' : result
    }
    
# transform(0)
