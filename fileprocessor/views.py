from django.shortcuts import render
from .forms import FileProcessForm
from .utils import process_file

def file_process_view(request):
    result = None
    warning = None
    if request.method == 'POST':
        form = FileProcessForm(request.POST, request.FILES)
        if form.is_valid():
            input_file = form.cleaned_data['input_file']
            mode = form.cleaned_data['mode']
            output = form.cleaned_data['output']
            api_key = form.cleaned_data['api_key']
            output_name = form.cleaned_data['output_name']
            pdf_method = form.cleaned_data.get('pdf_method')  # Get pdf_method if present
            import os
            if hasattr(input_file, 'temporary_file_path'):
                input_dir = os.path.dirname(input_file.temporary_file_path())
            else:
                input_dir = '/tmp'
            output_file_path = os.path.join(input_dir, output_name)
            # Pass pdf_method to process_file
            result = process_file(input_file, mode, output, api_key, output_file_path, pdf_method)
            if not api_key:
                warning = _("No API key provided. Only conversion to text file and PNGs of photos is available.")
    else:
        form = FileProcessForm()
    return render(request, 'fileprocessor/form.html', {'form': form, 'result': result, 'warning': warning})