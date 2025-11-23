from django import forms
from django.utils.translation import gettext_lazy as _

class FileProcessForm(forms.Form):
    input_file = forms.FileField(label=_("Input File (PDF or DOCX)"))
    mode = forms.ChoiceField(
        choices=[('integrated', _("Integrated")), ('separate', _("Separate"))],
        label=_("Image Description Mode"))
    output = forms.ChoiceField(
        choices=[('audio', _("Audio")), ('text', _("Text"))], 
        label=_("Output Type"))
    api_key = forms.CharField(widget=forms.PasswordInput, 
                              label=_("API Key"), 
                              help_text=_("Enter your OpenAI API key."),
                              required=False)
                              
    output_name = forms.CharField(
        label=_("Output File Name"),
        help_text=_("The output file will be saved in the same folder as the uploaded input file.")
    )
    pdf_method = forms.ChoiceField(
        choices=[
            ('llm', _("LLM (better for scanned books, more expensive)")),
            ('classic', _("Classic (cheaper, for regular PDFs)"))
        ],
        label=_("PDF Processing Method"),
        required=False,
        help_text=_("Choose LLM for scanned PDFs (books), Classic for regular PDFs. Only available if PDF and API key are provided.")
    )

    def clean(self):
        cleaned_data = super().clean()
        api_key = cleaned_data.get('api_key')
        input_file = cleaned_data.get('input_file')
        pdf_method = cleaned_data.get('pdf_method')
        if not input_file:
            self.add_error('input_file', _("Input file is required."))
        if input_file and input_file.name.lower().endswith('.pdf') and api_key:
            if not pdf_method:
                self.add_error('pdf_method', _("Please select a PDF processing method."))
        else:
            cleaned_data['pdf_method'] = None  # Hide/ignore if not PDF or no API key