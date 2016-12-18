from django.shortcuts import render

from forms import QueryImageForm

from process_image.run import run

# Create your views here.

def query_image(request):
    if request.method == 'POST':
        form = QueryImageForm(request.POST, request.FILES)
        if form.is_valid():
            result = run(request.FILES['file'])            
            return render(request, 'result.html', result)
        print form.errors
    else:
        form = QueryImageForm()
    return render(request, 'upload.html', {'form': form})
