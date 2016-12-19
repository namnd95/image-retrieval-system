from django import forms

class QueryImageForm(forms.Form):
    file = forms.ImageField()

