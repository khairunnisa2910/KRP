#impor library yang diperlukan

#implementasi web
from flask import Flask, render_template, request, url_for, redirect, session, g, jsonify
from flask_session import Session
from functools import wraps
import os

#visualisasi data
import plotly
import plotly.express as px

#pengolahan data
import pandas as pd
import numpy as np

#koneksi sql dengan python
import sqlite3
from datetime import datetime

#kirim data dari python ke web
import json

#membagi data
from sklearn.model_selection import train_test_split
#normalisasi data
from sklearn.preprocessing import MinMaxScaler
#knn
from sklearn.neighbors import KNeighborsClassifier
#k-fold
from sklearn.model_selection import cross_val_score
#confusion matrix
from sklearn.metrics import confusion_matrix

# manual knn
from manual_knn import KNearestNeighbors

#menghitung performa metode
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# mendefenisikan program flask
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

#sql

# Koneksi ke database
def connect_db():
	sql = sqlite3.connect('krp_knn.db', timeout=10)
	sql.row_factory = sqlite3.Row
	return sql

def get_db():
	if not hasattr(g, 'sqlite_db'):
		g.sqlite_db = connect_db()
	return g.sqlite_db

@app.teardown_appcontext
def close_db(error):
	if hasattr(g, 'sqlite_db'):
		g.sqlite_db.close()
  
def fetch_data_from_table(table_name):
    db = get_db()
    data_cur = db.execute(f'SELECT * FROM {table_name} order by id')
    data = data_cur.fetchall()
    columns = [desc[0] for desc in data_cur.description if desc[0].lower() != 'id']
    
    formatted_data = []
    for index, row in enumerate(data):
        row_dict = dict(row)
        formatted_data.append(row_dict)

    return formatted_data, columns

# dekorator untuk check admin
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    is_admin = 'admin' in session
    return render_template("index.html", title='KRP-KNN', is_admin = is_admin)

#  login
@app.route("/login", methods=["GET","POST"])
def login():
    
    if 'admin' in session:
        return redirect(url_for("dashboard"))
    
    if request.method == 'POST':
        
        username = request.form["username"]
        password = request.form["password"]
        
        if not username or not password:
            return render_template("login.html",err="Masukkan data login!", title='Login')
        
        else:
            db = get_db()
            user_cur = db.execute('select * from auth where username = ?', [username])
            myresult = user_cur.fetchone()
            
            if myresult is None:
                return render_template("login.html",err="Tidak ditemukan user tersebut!", title='Login')
            
            if myresult['username']==username and myresult['password']==password:
                session["admin"]=True
                session["username"] = myresult['username']
                session["nama"] = myresult['name']                    
                return redirect(url_for("dashboard"))
            else:
                return render_template("login.html",err="Password yang dimasukkan salah!", title='Login')

    return render_template("login.html", title='Login')

@app.route("/keluar")
@admin_required
def keluar():
    session.pop("admin",None)
    return redirect(url_for("index"))

# halaman dashboard
@app.route('/dashboard')
@admin_required
def dashboard():
    return render_template("dashboard.html", title='Homepage')

# halaman dataset
@app.route('/dataset')
@admin_required
def dataset():    
    
    db = get_db()
    data_col = db.execute("PRAGMA table_info(dataset)")
    columns_info = data_col.fetchall()
    column_names = {column['name']: column['type'] for column in columns_info if column['name'].lower() != 'id'}
        
    arr = fetch_data_from_table('dataset')[0]
    
    return render_template('dataset.html',data=arr, columns=column_names, title='Dataset')

# proses import dataset dan membuat kolom secara otomatis
@app.route('/importdataset', methods=['GET','POST'])
@admin_required
def importdataset():
    if request.method == 'POST':
        
        file = request.files.get('file')
        
        if not file:
            return redirect(url_for('dataset'))
        
        excel = pd.read_excel(file)
        type_data = excel.dtypes
        
        for column, dtype in excel.dtypes.items():
            if str(dtype).startswith('int'):
                type_data[column] = 'INTEGER'
            elif str(dtype).startswith('float'):
                type_data[column] = 'REAL'
            else:
                type_data[column] = 'TEXT'
                
        db = get_db()

        # hapus tabel
        table = 'dataset'
        drop_query = f"DROP TABLE IF EXISTS {table}"

        # buat tabel
        columns = excel.columns.tolist()
        columns_str = ', '.join([f"{col} {dtype}" for col, dtype in type_data.items()])
        create_query = f"CREATE TABLE {table} ({columns_str})"

        # insert data
        placeholders = ', '.join(['?' for _ in columns])
        insert_sql = f"INSERT INTO dataset ({', '.join(columns)}) VALUES ({placeholders})"
        data_tuples = [tuple(row) for _, row in excel.iterrows()]

        with db:
            db.execute(drop_query)
            db.execute(create_query)
            db.executemany(insert_sql, data_tuples)
        
        return redirect(url_for("dataset"))

# proses menambah data
@app.route('/tambah_data', methods=['POST'])
def tambah_data():
    if request.method == 'POST':
        
        data = request.form
        keys = list(data.keys())
        values = list(data.values())
        
        query = f"INSERT INTO dataset ({', '.join(keys)}) VALUES ({', '.join(['?'] * len(keys))})"
        db = get_db()
        db.execute(query, values)
        db.commit()
            
        response = jsonify({'message': 'Data berhasil ditambah'})
        return response, 200

    return redirect(url_for('dataset'))

# proses mengupdate data
@app.route('/update_data', methods=["POST"])
@admin_required
def update_data():    
    if request.method == 'POST':
        
        data = request.form
        data_update = {k: v for k, v in data.items() if k != 'id'}
        
        db = get_db()
        update_sql = "UPDATE dataset SET " + ", ".join([f"{key} = ?" for key in data_update.keys()]) + " WHERE id = ?"        
        db.execute(update_sql, (*data_update.values(), data['id']))
        db.commit()
            
        response = jsonify({'message': 'Data berhasil diubah'})
        return response, 200

    return redirect(url_for('dataset'))

# proses menghapus data
@app.route("/delete_data", methods=["POST"])
@admin_required
def delete_data():
    if request.method == 'POST':
         
        id = request.form.get('id')
        
        db = get_db()
        db.execute('DELETE FROM dataset WHERE id = ?', (id,))
        db.commit()

        response = jsonify({'message': 'Data berhasil dihapus'})
        return response, 200

# proses perhitungan rasio
@app.route('/rasio_data', methods=['GET', 'POST'])
@admin_required
def rasio_data():
    
    db = get_db()
    
    # select kolom
    data_col = db.execute("PRAGMA table_info(dataset)")
    columns_info = data_col.fetchall()
    column_names = [column['name'] for column in columns_info if column['name'].lower() != 'id' and column['type'] != 'TEXT']
        
    # kolom rasio_data
    rasio_col = db.execute("PRAGMA table_info(rasio_data)")
    columns_rasio = rasio_col.fetchall()
    col_select = [column['name'] for column in columns_rasio if column['name'].lower() != 'id']
    
    numerators = [
    'luas_lahan_baku_sawah_m2',
    'jumlah_sarana_prasarana_penyedia_pangan',
    'jumlah_pddk_tingkat_kesejahteraan_rendah',
    'desa_yang_tidak_memiliki_akses_yang_memadai',
    'jumlah_rt_tanpa_akses_air_bersih',
    'jumlah_penduduk'
    ]

    denominators = [
        'jumlah_penduduk',
        'jumlah_rumah_tangga',
        'jumlah_penduduk',
        'dibagi_1',
        'jumlah_rumah_tangga',
        'jumlah_nakes'
    ]
    
    arr = fetch_data_from_table('rasio_data')[0]
    arr2 = fetch_data_from_table('dataset')[0]

    if request.method == 'POST':
        
        rasio_results = []

        for num, denom in zip(numerators, denominators):
            
            numerator_data = db.execute(f"SELECT {num}, id FROM dataset").fetchall()
            if denom == 'dibagi_1' :
                denominator_data = db.execute("SELECT 1, id FROM dataset").fetchall()
            else:
                denominator_data = db.execute(f"SELECT {denom}, id FROM dataset").fetchall()
            
            name = ('rasio_' + num + '_per_' + denom) if denom != 'dibagi_1' else num

            for n, d in zip(numerator_data, denominator_data):
                id_ = d[1]
                
                # ambil id yang cocok dengan data
                existing_entry = next((item for item in rasio_results if item['id'] == id_), None)
                
                if not existing_entry:
                    new_entry = {'id': id_}
                    rasio_results.append(new_entry)
                    existing_entry = new_entry
                try:
                    rasio = n[0] / d[0]
                except ZeroDivisionError:
                    rasio = 0
                    
                existing_entry[name] = rasio
        
        # hapus tabel jika ada
        drop_query = "DROP TABLE IF EXISTS rasio_data"
        db.execute(drop_query)

        # buat dan defenisikan tabel
        column_definitions = ["id INTEGER"] 
        for num, denom in zip(numerators, denominators):
            column_name = ('rasio_' + num + '_per_' + denom) if denom != 'dibagi_1' else num
            column_definitions.append(f"{column_name} REAL")
        create_query = f"CREATE TABLE rasio_data ({', '.join(column_definitions)})"
        db.execute(create_query)
        
        # insert data
        placeholders = ', '.join(['?' for _ in column_definitions])
        insert_query = f"INSERT INTO rasio_data VALUES ({placeholders})"
        data_to_insert = []
        column_names = [col.split(' ')[0] for col in column_definitions]
        for entry in rasio_results:
            row = [entry.get(col, None) for col in column_names] 
            data_to_insert.append(tuple(row))
        
        db.executemany(insert_query, data_to_insert)
        db.commit()
                        
        return redirect(url_for('rasio_data'))
    
    id_to_desa = {item['id']: item['desa'] for item in arr2}
    
    for item in arr:
        item['desa'] = id_to_desa.get(item['id'], None)
    
    return render_template('rasio.html', columns=column_names, col_select=col_select, title='Rasio Data', rasio=arr)

# halaman split data
@app.route('/split', methods=["GET", "POST"])
@admin_required
def split():
    
    myresult, column_names = fetch_data_from_table('rasio_data')
    myresult_dataset = fetch_data_from_table('dataset')[0]
    train = fetch_data_from_table('datatraining')[0]
    test = fetch_data_from_table('datatesting')[0]
        
    if request.method == 'POST':
        
        split_ratio = float(request.form['rasio_split'])
        
        if myresult:
            myresult = pd.DataFrame(myresult)
            myresult_dataset = pd.DataFrame(myresult_dataset)
            
            myresult['aktual'] = myresult_dataset.iloc[:, -1]

            X = myresult.iloc[:, :-1]
            y = myresult.iloc[:, -1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42, stratify=y)
            
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            columns = train_data.columns.to_list()
            
            dynamic_columns = ', '.join(columns)
            placeholders = ', '.join(['?' for _ in columns])
            
            db = get_db()
            for k, v in {'datatraining': train_data, 'datatesting': test_data}.items():
                db.execute(f"DROP TABLE IF EXISTS {k}")

                create_table_sql = f"CREATE TABLE {k} ({dynamic_columns})"
                db.execute(create_table_sql)
                
                insert_sql = f"INSERT INTO {k} ({', '.join(columns)}) VALUES ({placeholders})"
                data_tuples = [tuple(row) for _, row in v.iterrows()]
                db.executemany(insert_sql, data_tuples)
            
            if str(myresult['aktual'].dtype).startswith('int'):
                prediksi_dtype = 'INTEGER'
            elif str(myresult['aktual'].dtype).startswith('float'):
                prediksi_dtype = 'REAL'
            else:
                prediksi_dtype = 'TEXT'
            
            alter_table_sql = f"ALTER TABLE datatesting ADD COLUMN prediksi {prediksi_dtype}"
            db.execute(alter_table_sql)
            db.commit()
            
        return redirect(url_for('split'))
    
    id_to_desa = {item['id']: item['desa'] for item in myresult_dataset}
    
    for item in train:
        item['desa'] = id_to_desa.get(item['id'], None)
    
    for item in test:
        item['desa'] = id_to_desa.get(item['id'], None)
    
    return render_template("split.html", data_train=train, data_test=test, columns=column_names, title='Split Data')

# normalisasi data
@app.route("/normalisasi_data", methods=["GET","POST"])
@admin_required
def normalisasi_data():
    
    train, column_names = fetch_data_from_table('normalisasi_train')
    test = fetch_data_from_table('normalisasi_test')[0]
    
    arr = fetch_data_from_table('dataset')[0]
    id_to_desa = {item['id']: item['desa'] for item in arr}
    
    for item in train:
        item['desa'] = id_to_desa.get(item['id'], None)
    
    for item in test:
        item['desa'] = id_to_desa.get(item['id'], None)

    if request.method == "POST" :
        train = pd.DataFrame(fetch_data_from_table('datatraining')[0])
        test = pd.DataFrame(fetch_data_from_table('datatesting')[0])
        
        # ambil selain pertama dan terakhir
        X_train = train.iloc[:, 1:-1]
        X_test = test.iloc[:, 1:-2]
                
        column_names = train.columns.to_list()
        columns = X_train.columns.to_list()
        
        min_max_scaler = MinMaxScaler(feature_range=(0,1)) #inisialisasi normalisasi MinMax
        data_train = min_max_scaler.fit_transform(X_train)
        data_test = min_max_scaler.transform(X_test)
        
        train[columns] = data_train
        test[columns] = data_test
            
        dynamic_columns = ', '.join([f'"{col}" INTEGER' if col == 'id' else f'"{col}" TEXT' for col in column_names])
        placeholders = ', '.join(['?' for _ in column_names])
        
        db = get_db()

        for k, v in {'normalisasi_train': train[column_names], 'normalisasi_test': test[column_names], 'log_satu_data_testing': test[column_names]}.items():
            db.execute(f"DROP TABLE IF EXISTS {k}")

            create_table_sql = f"CREATE TABLE {k} ({dynamic_columns})"
            db.execute(create_table_sql)
            
            if k != 'log_satu_data_testing':
                insert_sql = f"INSERT INTO {k} ({', '.join(column_names)}) VALUES ({placeholders})"
                data_tuples = [tuple(row) for _, row in v.iterrows()]
                db.executemany(insert_sql, data_tuples)
            
        if str(train['aktual'].dtype).startswith('int'):
            prediksi_dtype = 'INTEGER'
        elif str(train['aktual'].dtype).startswith('float'):
            prediksi_dtype = 'REAL'
        else:
            prediksi_dtype = 'TEXT'
            
        alter_table_sql = f"ALTER TABLE normalisasi_test ADD COLUMN prediksi {prediksi_dtype}"
        db.execute(alter_table_sql)
        
        alter_table_sql = f"ALTER TABLE log_satu_data_testing ADD COLUMN prediksi {prediksi_dtype}"
        db.execute(alter_table_sql)
        
        alter_table_sql = "ALTER TABLE log_satu_data_testing ADD COLUMN request_date TEXT"
        db.execute(alter_table_sql)
        
        alter_table_sql = "ALTER TABLE log_satu_data_testing ADD COLUMN desa TEXT"
        db.execute(alter_table_sql)
        
        alter_table_sql = "ALTER TABLE log_satu_data_testing DROP COLUMN aktual"
        db.execute(alter_table_sql)
        
        alter_table_sql = "ALTER TABLE log_satu_data_testing DROP COLUMN id"
        db.execute(alter_table_sql)
            
        db.commit()
    
    return render_template("normalisasi.html", data_train=train, data_test=test, columns=column_names, title='Normalisasi Data')

@app.route('/klasifikasi_knn', methods=['GET','POST'])
@admin_required
def klasifikasi_knn():
    
    train = pd.DataFrame(fetch_data_from_table('normalisasi_train')[0])
    test, columns = fetch_data_from_table('normalisasi_test')
    test = pd.DataFrame(test)
    test2 = test.drop('prediksi', axis=1)
    
    arr = fetch_data_from_table('dataset')[0]

    # check nilai k
    db = get_db()
    check_nilai_k = db.execute('select * from nilai_k')
    k_val = check_nilai_k.fetchone()

    if k_val['k']:
        angka_k = k_val['k']
    else:
        angka_k = False
    
    try:
        # check akurasi
        true_y = test2.iloc[:, -1]
        pred_y = test.iloc[:, -1]
        true_y = true_y.astype(str)
        pred_y = pred_y.astype(str)

        akurasi = round(accuracy_score(true_y, pred_y) * 100, 2)
    
    except Exception:
        akurasi = 0

    if request.method == 'POST':
        
        nilai_k = int(request.form['k_value'])
        db = get_db()
        cursor = db.execute('UPDATE nilai_k SET k = ? WHERE id = 1', [nilai_k])
        if cursor.rowcount == 0:
            db.execute('INSERT INTO nilai_k(id, k) VALUES (1, ?)', [nilai_k])
        db.commit()

        # ambil selain pertama dan terakhir
        X_train = train.iloc[:, 1:-1]
        y_train = train.iloc[:, -1]
        
        X_test = test2.iloc[:, 1:-1]
        y_test = test2.iloc[:, -1]
        
        # klasifikasi menggunakan KNN
        model = KNeighborsClassifier(n_neighbors=nilai_k)
        model.fit(X_train, y_train)
        result = model.predict(X_test)
    
        test['prediksi'] = result
        
        # Memasukkan data ke dalam tabel datatesting
        db = get_db()
        for _, row in test.iterrows():
            db.execute('UPDATE normalisasi_test SET prediksi = ? WHERE id = ?', [row['prediksi'], row['id']])
        
        db.commit()

        akurasi = round(accuracy_score(y_test,result)*100,2)
        
        return redirect(url_for('klasifikasi_knn'))
    
    formatted_data = test.reset_index().to_dict(orient='records')
    
    id_to_desa = {item['id']: item['desa'] for item in arr}
    
    for item in formatted_data:
        item['desa'] = id_to_desa.get(item['id'], None)
    
    return render_template("klasifikasi_knn.html", test=formatted_data, akurasi=akurasi, columns=columns, nilai_k=angka_k, title='Klasifikasi KNN')

# find optimal value
@app.route("/elbow", methods=["GET","POST"])
@admin_required
def elbow():

    train = pd.DataFrame(fetch_data_from_table('normalisasi_train')[0])
    test = pd.DataFrame(fetch_data_from_table('normalisasi_test')[0])
    test2 = test.drop('prediksi', axis=1)
        
    # ambil selain pertama dan terakhir
    X_train = train.iloc[:, 1:-1]
    y_train = train.iloc[:, -1]
    
    X_test = test2.iloc[:, 1:-1]
    y_test = test2.iloc[:, -1]
    
    nilai_k = range(1,31)
    
    error_rate = []
    for i in nilai_k:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    # Generate plot
    fig = px.line(x=nilai_k, y=error_rate, markers=True).update_layout(
        xaxis_title='Nilai K', yaxis_title='Error Rate')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template("plot.html", image=graphJSON, title='Elbow - Find Optimal Value')

@app.route('/score')
@admin_required
def score():

    train = pd.DataFrame(fetch_data_from_table('normalisasi_train')[0])
    test = pd.DataFrame(fetch_data_from_table('normalisasi_test')[0])
    test2 = test.drop('prediksi', axis=1)

    y_test = test2.iloc[:, -1]
    y_pred = test.iloc[:, -1]
    
    y_test = y_test.astype(str).astype(float).astype(int)
    y_pred = y_pred.astype(int)
    
    y_train = train.iloc[:, -1]
    y_train = y_train.astype(str).astype(float).astype(int)
    label = sorted(y_train.unique())
        
    matrix = confusion_matrix(y_test,y_pred, labels=label) 

    akurasi = accuracy_score(y_test, y_pred)

    presisi= precision_score(y_test, y_pred, labels=label, average='macro')

    rikol = recall_score(y_test, y_pred, labels=label, average='macro')

    efwan = f1_score(y_test, y_pred, labels=label,  average='macro')

    performa = pd.DataFrame({
                'name' : ['Akurasi','Precision','Recall','F1-Score'],
                'val' : [akurasi, presisi, rikol, efwan] 
                }).sort_values(ascending=False, by='val')


    performa = pd.DataFrame({
                'name' : ['Akurasi','Precision','Recall','F1-Score'],
                'val' : [akurasi, presisi, rikol, efwan] 
                }).sort_values(ascending=False, by='val')


    # Generate plot
    fig = px.bar(performa, x='name', y='val', color='name').update_layout(
        xaxis_title='Evaluasi', yaxis_title='Nilai')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template("score.html", matrix=matrix, title='Evaluasi', label=label, image=graphJSON)

@app.route("/sebaran_data", methods=['GET'])
@admin_required
def plot_sebaran():

    df = pd.DataFrame(fetch_data_from_table('dataset')[0])
    kolom_numeric = df.select_dtypes(include=['int64']).drop('id', axis=1)
    kolom_text = df.select_dtypes(include=['object'])
    
    graphJSON = []

    for col_name in kolom_numeric.columns:
        fig = None
        grafik_title = col_name.replace('_', ' ').title()
        if col_name in kolom_text:
            fig = px.bar(df, x=col_name, color=kolom_numeric.columns[-1], title=f'Grafik {grafik_title}', labels={col_name:grafik_title, 'count':'Jumlah'})
        else:
            fig = px.histogram(df, x=col_name, title=f'Sebaran Data {grafik_title}', labels={col_name:grafik_title, 'count':'Jumlah'})
        
        graphJSON.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    
    return render_template("plot2.html", title='Sebaran Data', image=graphJSON)

@app.route("/sebaran_prioritas", methods=['GET'])
@admin_required
def plot_sebaran_prioritas():

    df = pd.DataFrame(fetch_data_from_table('dataset')[0])
    kolom_numeric = df.select_dtypes(include=['int64']).drop('id', axis=1)
    kolom_text = df.select_dtypes(include=['object'])

    prioritas_dict = {
        1: 'Sangat Rentan Pangan',
        2: 'Rentan Pangan',
        3: 'Agak Rentan Pangan',
        4: 'Agak Tahan Pangan',
        5: 'Tahan Pangan',
        6: 'Sangat Tahan Pangan'
    }

    df['priority_map'] = df['prioritas'].map(prioritas_dict)
    
    graphJSON = []

    for col_name in kolom_numeric.columns:
        data_count = df.groupby('priority_map')[col_name].size().reset_index(name='count')
        grafik_title = col_name.replace('_', ' ').title()
        title = f'Jumlah Data {grafik_title}'
        fig = px.bar(data_count, x='priority_map', y='count', color='priority_map', title=title, labels={'priority_map':'Prioritas', 'count':'Jumlah'})
        fig.update_layout(showlegend=False)
        graphJSON.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    
    return render_template("plot2.html", title='Sebaran Data Berdasarkan Prioritas', image=graphJSON)

# detail
@app.route("/detail/<int:id>", methods=["GET"])
@admin_required
def detail(id):
    db = get_db()
    check_nilai_k = db.execute('select * from nilai_k')
    k_val = check_nilai_k.fetchone()
    
    # Fetch the training data from the database    
    df = pd.DataFrame(fetch_data_from_table('normalisasi_train')[0])
    
    # Drop kolom aktual dan id
    y_train = df.pop('aktual').tolist()
    y_id = df.pop('id').tolist()
    
    arr = fetch_data_from_table('dataset')[0]
    id_to_desa = {item['id']: item['desa'] for item in arr}
    id_to_kelas = {item['id']: item['prioritas'] for item in arr}
        
    # Get column names
    col_name = df.columns.tolist()
    
    # Get training array
    df = df.astype(float)
    X_train = df.values.tolist()
    
    # Fetch the test data based on the provided 'id' parameter
    test = db.execute("SELECT * FROM normalisasi_test WHERE id = ?", [id]).fetchone()

    X_test = [float(test[x]) for x in col_name]
    
    desa_test = id_to_desa.get(test['id'], None)
    
    # Create an instance of KNearestNeighbors and fit it with the training data
    knn = KNearestNeighbors(k=k_val['k'])
    knn.fit(X_train, y_train)

    # Get the detailed prediction for the test point
    prediction = knn.detailed_predict(X_test, y_id)
    
    # Menyiapkan data untuk dikirim ke template
    data_testing = [desa_test] + prediction['point']

    k_nearest_details = [
        {
            'desa': id_to_desa.get(detail['id'], None), 
            'details': detail, 
            'kelas': id_to_kelas.get(detail['id'], None)
        } 
        for i, (detail, _) in enumerate(prediction['k_nearest'])
        ]
    
    distances_details = [
        {
            'desa': id_to_desa.get(detail['id'], None), 
            'details': detail, 
            'kelas': id_to_kelas.get(detail['id'], None)
        } 
        for i, (detail, _) in enumerate(prediction['distances'])
    ]

    
    context = {
        'data_testing': data_testing,
        'nilai_k': prediction['k'],
        'col_name': ['desa'] + col_name,
        'vote_result': prediction['vote_result'],
        'distances_details': sorted(distances_details, key=lambda x: x['details']['id']),
        'k_nearest_details': k_nearest_details,
    }

    
    return render_template("detail.html", context=context, title='Detail Perhitungan')



# halaman dashboard
@app.route('/uji1')
@admin_required
def uji1():
    db = get_db()
    columns_info = db.execute("PRAGMA table_info(dataset)").fetchall()
    column_names = {column['name']: column['type'] for index, column in enumerate(columns_info) if column['name'].lower() != 'id' and index != len(columns_info) - 1}
    return render_template("uji1.html", title='Uji 1', columns=column_names,)

# api uji 1 data
@app.route('/api/uji1', methods=['POST'])
def api_uji():
    
    db = get_db()
    check_nilai_k = db.execute('select * from nilai_k')
    k_val = check_nilai_k.fetchone()
    train = pd.DataFrame(fetch_data_from_table('normalisasi_train')[0])
    
    desa = request.form.get('desa')
    luas_lahan_baku_sawah_m2 = request.form.get('luas_lahan_baku_sawah_m2')
    jumlah_sarana_prasarana_penyedia_pangan = request.form.get('jumlah_sarana_prasarana_penyedia_pangan')
    jumlah_pddk_tingkat_kesejahteraan_rendah = request.form.get('jumlah_pddk_tingkat_kesejahteraan_rendah')
    desa_yang_tidak_memiliki_akses_yang_memadai = request.form.get('desa_yang_tidak_memiliki_akses_yang_memadai')
    jumlah_rt_tanpa_akses_air_bersih = request.form.get('jumlah_rt_tanpa_akses_air_bersih')
    jumlah_nakes = request.form.get('jumlah_nakes')
    jumlah_penduduk = request.form.get('jumlah_penduduk')
    jumlah_rumah_tangga= request.form.get('jumlah_rumah_tangga')
    
    fields = [desa, luas_lahan_baku_sawah_m2, jumlah_sarana_prasarana_penyedia_pangan, 
              jumlah_pddk_tingkat_kesejahteraan_rendah, desa_yang_tidak_memiliki_akses_yang_memadai,
              jumlah_rt_tanpa_akses_air_bersih, jumlah_nakes, jumlah_penduduk, jumlah_rumah_tangga]

    for field in fields:
        if not field or field.strip() == "":
            return jsonify({'message': 'Semua field wajib diisi!'}), 400
    
    # rasio
    numerators = [
    'luas_lahan_baku_sawah_m2',
    'jumlah_sarana_prasarana_penyedia_pangan',
    'jumlah_pddk_tingkat_kesejahteraan_rendah',
    'desa_yang_tidak_memiliki_akses_yang_memadai',
    'jumlah_rt_tanpa_akses_air_bersih',
    'jumlah_penduduk'
    ]

    denominators = [
       
        'jumlah_penduduk',
        'jumlah_rumah_tangga',
        'jumlah_penduduk',
        'dibagi_1',
        'jumlah_rumah_tangga',
        'jumlah_nakes'
    ]
    
    data = {
        'desa': desa,
        'luas_lahan_baku_sawah_m2': float(luas_lahan_baku_sawah_m2),
        'jumlah_sarana_prasarana_penyedia_pangan': float(jumlah_sarana_prasarana_penyedia_pangan),
        'jumlah_pddk_tingkat_kesejahteraan_rendah': float(jumlah_pddk_tingkat_kesejahteraan_rendah),
        'desa_yang_tidak_memiliki_akses_yang_memadai': float(desa_yang_tidak_memiliki_akses_yang_memadai),
        'jumlah_rt_tanpa_akses_air_bersih': float(jumlah_rt_tanpa_akses_air_bersih),
        'jumlah_nakes': float(jumlah_nakes),
        'jumlah_penduduk': float(jumlah_penduduk),
        'jumlah_rumah_tangga': float(jumlah_rumah_tangga)
    }
    
    to_ratio = {}
    
    for num, denom in zip(numerators, denominators):          
        if denom == 'dibagi_1':
            res = data[num]
        else:
            res = data[num] / data[denom]
        
        name = ('rasio_' + num + '_per_' + denom) if denom != 'dibagi_1' else num
        
        to_ratio[name] = res
    
    X_train = train.iloc[:, 1:-1]
    y_train = train.iloc[:, -1]    
    X_test = pd.DataFrame(to_ratio, index=[0])

    
    # klasifikasi menggunakan KNN
    model = KNeighborsClassifier(n_neighbors=k_val['k'])
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    
    result_to_image = int(float(result[0]))
    
    X_test['prediksi'] = result
    X_test['desa'] = desa
    X_test['request_date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    keys = X_test.columns
    values = X_test.values[0]
    
    query = f"INSERT INTO log_satu_data_testing ({', '.join(keys)}) VALUES ({', '.join(['?'] * len(keys))})"
    db = get_db()
    db.execute(query, values)
    db.commit()
    
    prioritas_dict = {
        1: 'Prioritas 1 (Sangat Rentan Pangan)',
        2: 'Prioritas 2 (Rentan Pangan)',
        3: 'Prioritas 3 (Agak Rentan Pangan)',
        4: 'Prioritas 4 (Agak Tahan Pangan)',
        5: 'Prioritas 5 (Tahan Pangan)',
        6: 'Prioritas 6 (Sangat Tahan Pangan)'
    }
    
    priority = prioritas_dict.get(result_to_image, 'Prioritas tidak valid')

    response = jsonify({'desa': desa, 'priority' : priority})
    return response, 200

@app.route('/password', methods=['GET','POST'])
@admin_required
def password():   
    if request.method == "POST":
        
        pass1 = request.form['pass1']
        pass2 = request.form['pass2']
        pass3 =request.form['pass3']

        cursor = get_db()
        user_cek = cursor.execute("SELECT * FROM auth WHERE username = ? AND password = ?", (session['username'], pass1))
        result = user_cek.fetchone()

        if pass1 == '' or pass2 == '' or pass3 == '':
            message = 'Field bertanda * harus diisi.'
        elif result is None:
            message = 'Password lama salah.'
        elif pass2 != pass3:
            message = 'Password baru dan konfirmasi password baru tidak sama.'
        else:
            cursor.execute("UPDATE auth SET password=? WHERE username=?", (pass2, session['username']))
            cursor.commit()
            message = 'Password berhasil diubah.'

        return render_template('password.html', title='Password', message=message)

    if request.method == 'GET':
        return render_template('password.html', title='Password')

@app.route('/nama', methods=['GET','POST'])
@admin_required
def nama():
    if request.method == "POST":
    
        nama = request.form['nama']
        cursor = get_db()
        cursor.execute("UPDATE auth SET name=? WHERE username=?", (nama, session["username"]))
        cursor.commit()

        message = 'Nama Berhasil di Ubah! Silahkan Login Kembali!'

        return render_template('nama.html', title='Ubah Profil', message=message)

    if request.method == 'GET':
        return render_template('nama.html', title='Ubah Profil')   

if __name__=='__main__':
    app.run(debug=True, port=5005)