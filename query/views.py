from django.shortcuts import render

from forms import QueryImageForm

import process_query_image

# Create your views here.

def query_image(request):
    if request.method == 'POST':
        form = QueryImageForm(request.POST, request.FILES)
        if form.is_valid():
            result = process_query_image.run(request.FILES['file'])
            context = {
                'result': result
            }
            return render(request, 'result.html', context)
        print form.errors
    else:
        form = QueryImageForm()
    return render(request, 'upload.html', {'form': form})
