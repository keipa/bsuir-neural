from django.shortcuts import render
from django.http import JsonResponse
import web.BL.predict

def index(request):
    return render(request, 'web/index.html', {})


def predict(request):
    data = {"therapy" : "Please generate SMILES"}
    if request.GET.get('smiles', None) != str():
        output = web.BL.predict.predict(request.GET.get('smiles', None))
        data["therapy"] = output
    return JsonResponse(data)
