import numpy as np
import plotly.utils
from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio


#pio.renderers.default = 'browser'

"""     ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
"""

app = Flask(__name__)

urunler = ['kıyma', 'tavuk', 'hindi', 'Bakla-kuru', 'Bakla-taze', 'Bamya',
               'Barbunya-kuru', 'Bezelye-taze', 'Biber-yeşil', 'Brüksel laha.',
               'Domates', 'Enginar', 'Fasulye-kuru', 'Fasulye-taze', 'Havuç',
               'Salatalık', 'Ispanak', 'Kabak', 'Karnabahar', 'Kereviz-baş',
               'Kırmızı-pancar', 'Lahana', 'Mantar', 'Marul', 'Mercimek-kuru',
               'Nohut', 'Pancar', 'Patates', 'Patlıcan', 'Pazı', 'Pırasa',
               'Semizotu', 'Buğday ekmeği', 'Bulgur', 'Erişte', 'Makarna',
               'Mısır', 'Mısır unu', 'Nişasta', 'Pilav', 'Pirinç unu', 'Şehriye',
               'Tarhana', 'Yulaf unu', 'yumurta', 'beyaz peynir', 'Kaşar peyniri',
               'krema', 'yoğurt', 'sucuk', 'salam', 'sosis', 'tam buğday ekmek',
               'kaymak', 'bal', 'tereyağ', 'Armut', 'Çilek', 'Elma',
               'Erik türleri', 'İncir', 'Karadut', 'Karpuz', 'Kavun', 'Kayısı',
               'Kiraz', 'Limon', 'Mandalina', 'Muz', 'Nar', 'Portakal', 'Şeftali',
               'Üzüm', 'Vişne', 'fındık', 'fıstık', 'kaju', 'badem', 'ceviz']


@app.route('/')
def home():
    return render_template('index.html', urunler=urunler)


@app.route('/',methods=['GET', 'POST'])
def predict():

    product_name = request.form['urun']
    #products_name = [x for x in request.form.values()]
    print(product_name)
    print('######################################')

    results = pd.read_csv('data/predictions.csv', index_col='tarih')
    real = pd.read_csv('data/realdata.csv', index_col='tarih')
    results2 = results[results['ürün'] == product_name]
    temp = real[real["ürün"] == product_name]
    whole = temp.groupby(temp.index).mean()


    # set up plotly figure
    fig = go.Figure()

    # add line / trace 1 to figure
    fig.add_trace(go.Scatter(
        x=results2.index,
        y=results2['ürün fiyatı'],
        hovertext=results2['ürün fiyatı'],
        hoverinfo="text",
        name='Tahmin',
        marker=dict(
            color="red"
        ),
        showlegend=True
    ))

    # add line / trace 2 to figure
    fig.add_trace(go.Scatter(
        x=whole.index,
        y=whole['ürün fiyatı'],
        hovertext=whole['ürün fiyatı'],
        hoverinfo="text",
        name='Gerçek Değerler',
        marker=dict(
            color="green"
        ),
        showlegend=True
    ))

    fig.update_layout(
        title=f"{product_name} için Fiyat Tahmini",
        xaxis_title="Tarih",
        yaxis_title="Ürün Fiyatı (TL)",
        template="ggplot2",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ))

    #fig.show()
    fig.write_html('figure.html', auto_open=True)

    return render_template('index.html', urunler=urunler)



if __name__ == "__main__":
    app.run(debug=True)
