# -*- coding: utf-8 -*-
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
import json, uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'nexus-secret-key-9999'

PRODUCTS_FILE = 'products.json'
CART_FILE     = 'cart.json'
ORDERS_FILE   = 'orders.json'
WISHLIST_FILE = 'wishlist.json'

# ══════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════

def load_products():
    with open(PRODUCTS_FILE, encoding='utf-8') as f:
        return json.load(f)['products']

def load_cart():
    try:
        with open(CART_FILE, encoding='utf-8') as f:
            return json.load(f)['cart']
    except:
        return []

def save_cart(cart):
    with open(CART_FILE, 'w', encoding='utf-8') as f:
        json.dump({"cart": cart}, f, indent=4, ensure_ascii=False)

def load_orders():
    try:
        with open(ORDERS_FILE, encoding='utf-8') as f:
            return json.load(f)['orders']
    except:
        return []

def save_orders(orders):
    with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"orders": orders}, f, indent=4, ensure_ascii=False)

def load_wishlist():
    try:
        with open(WISHLIST_FILE, encoding='utf-8') as f:
            return json.load(f)['wishlist']
    except:
        return []

def save_wishlist(wl):
    with open(WISHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump({"wishlist": wl}, f, indent=4, ensure_ascii=False)

def get_product(products, pid):
    return next((p for p in products if p['id'] == pid), None)

def build_cart_summary(products, cart):
    enriched, subtotal = [], 0
    for item in cart:
        p = get_product(products, item['product_id'])
        if not p:
            continue
        disc  = p.get('discount', 0)
        price = round(p['price'] * (1 - disc/100), 2)
        line  = round(item['qty'] * price, 2)
        subtotal += line
        enriched.append({
            'product_id': item['product_id'],
            'name':       p['name'],
            'price':      price,
            'orig_price': p['price'],
            'qty':        item['qty'],
            'subtotal':   line,
            'icon':       p.get('icon', 'box'),
            'category':   p.get('category', 'General'),
            'badge':      p.get('badge', ''),
            'discount':   disc,
            'rating':     p.get('rating', 4.0),
        })
    gst         = round(subtotal * 0.18, 2)
    delivery    = 0 if subtotal >= 999 else 49
    grand_total = round(subtotal + gst + delivery, 2)
    saved       = sum((get_product(load_products(), i['product_id'])['price'] - e['price']) * e['qty']
                      for i, e in zip(cart, enriched) if get_product(load_products(), i['product_id']))
    return enriched, round(subtotal,2), gst, delivery, grand_total, delivery==0, max(0,999-subtotal), round(saved,2)

# ══════════════════════════════════════════
#  ROUTES — CATALOG
# ══════════════════════════════════════════

@app.route('/')
def index():
    products  = load_products()
    cart      = load_cart()
    wishlist  = load_wishlist()
    cart_count= sum(i['qty'] for i in cart)
    cart_ids  = {i['product_id'] for i in cart}
    wish_ids  = set(wishlist)
    sort_by   = request.args.get('sort','default')
    if sort_by == 'price_asc':   products = sorted(products, key=lambda p: p['price'])
    elif sort_by == 'price_desc':products = sorted(products, key=lambda p: p['price'], reverse=True)
    elif sort_by == 'rating':    products = sorted(products, key=lambda p: p.get('rating',0), reverse=True)
    elif sort_by == 'discount':  products = sorted(products, key=lambda p: p.get('discount',0), reverse=True)
    return render_template('index.html', products=products, cart_count=cart_count,
                           cart_ids=cart_ids, wish_ids=wish_ids, sort_by=sort_by)

# ══════════════════════════════════════════
#  ROUTES — CART
# ══════════════════════════════════════════

@app.route('/add/<int:pid>')
def add_to_cart(pid):
    products = load_products()
    cart     = load_cart()
    product  = get_product(products, pid)
    if not product:
        flash('Product not found.', 'error')
        return redirect(url_for('index'))
    for item in cart:
        if item['product_id'] == pid:
            item['qty'] += 1
            save_cart(cart)
            flash(f'{product["name"]} qty updated!', 'success')
            return redirect(request.referrer or url_for('cart_page'))
    cart.append({'product_id': pid, 'qty': 1})
    save_cart(cart)
    flash(f'{product["name"]} added to cart!', 'success')
    return redirect(request.referrer or url_for('cart_page'))

@app.route('/update/<int:pid>/<action>')
def update_cart(pid, action):
    cart = load_cart()
    for item in cart:
        if item['product_id'] == pid:
            if action == 'inc': item['qty'] += 1
            elif action == 'dec':
                item['qty'] -= 1
                if item['qty'] <= 0: cart.remove(item)
            break
    save_cart(cart)
    return redirect(url_for('cart_page'))

@app.route('/remove/<int:pid>')
def remove_from_cart(pid):
    save_cart([i for i in load_cart() if i['product_id'] != pid])
    return redirect(url_for('cart_page'))

@app.route('/clear')
def clear_cart():
    save_cart([])
    return redirect(url_for('cart_page'))

@app.route('/cart')
def cart_page():
    products = load_products()
    cart     = load_cart()
    enriched, subtotal, gst, delivery, grand_total, free_delivery, threshold, saved = \
        build_cart_summary(products, cart)
    return render_template('cart.html', cart=enriched, total=subtotal, gst=gst,
                           delivery=delivery, grand_total=grand_total,
                           free_delivery=free_delivery, threshold=int(threshold),
                           cart_count=sum(i['qty'] for i in enriched), saved=saved)

# ══════════════════════════════════════════
#  ROUTES — WISHLIST
# ══════════════════════════════════════════

@app.route('/wishlist/toggle/<int:pid>')
def toggle_wishlist(pid):
    wl = load_wishlist()
    if pid in wl: wl.remove(pid)
    else: wl.append(pid)
    save_wishlist(wl)
    return redirect(request.referrer or url_for('index'))

@app.route('/wishlist')
def wishlist_page():
    wl       = load_wishlist()
    products = load_products()
    items    = [p for p in products if p['id'] in wl]
    cart     = load_cart()
    cart_count = sum(i['qty'] for i in cart)
    cart_ids = {i['product_id'] for i in cart}
    return render_template('wishlist.html', items=items, cart_count=cart_count,
                           cart_ids=cart_ids, wish_ids=set(wl))

# ══════════════════════════════════════════
#  ROUTES — CHECKOUT
# ══════════════════════════════════════════

@app.route('/checkout', methods=['GET','POST'])
def checkout():
    products = load_products()
    cart     = load_cart()
    if not cart:
        flash('Your cart is empty!', 'error')
        return redirect(url_for('cart_page'))
    enriched, subtotal, gst, delivery, grand_total, free_delivery, threshold, saved = \
        build_cart_summary(products, cart)
    cart_count = sum(i['qty'] for i in enriched)
    return render_template('checkout.html', cart=enriched, total=subtotal, gst=gst,
                           delivery=delivery, grand_total=grand_total,
                           free_delivery=free_delivery, cart_count=cart_count, saved=saved)

@app.route('/place_order', methods=['POST'])
def place_order():
    products = load_products()
    cart     = load_cart()
    if not cart:
        return redirect(url_for('index'))

    enriched, subtotal, gst, delivery, grand_total, free_delivery, threshold, saved = \
        build_cart_summary(products, cart)

    name     = request.form.get('name','')
    phone    = request.form.get('phone','')
    address  = request.form.get('address','')
    pincode  = request.form.get('pincode','')
    city     = request.form.get('city','')
    state    = request.form.get('state','')
    payment  = request.form.get('payment','COD')

    order_id = 'NX-' + uuid.uuid4().hex[:8].upper()
    order = {
        'id':         order_id,
        'date':       datetime.now().strftime('%d %b %Y, %I:%M %p'),
        'timestamp':  datetime.now().isoformat(),
        'name':       name,
        'phone':      phone,
        'address':    f'{address}, {city}, {state} - {pincode}',
        'payment':    payment,
        'items':      enriched,
        'subtotal':   subtotal,
        'gst':        gst,
        'delivery':   delivery,
        'grand_total':grand_total,
        'saved':      saved,
        'status':     'Confirmed',
    }
    orders = load_orders()
    orders.insert(0, order)
    save_orders(orders)
    save_cart([])
    return redirect(url_for('order_success', order_id=order_id))

@app.route('/order/success/<order_id>')
def order_success(order_id):
    orders = load_orders()
    order  = next((o for o in orders if o['id'] == order_id), None)
    cart   = load_cart()
    cart_count = sum(i['qty'] for i in cart)
    return render_template('order_success.html', order=order, cart_count=cart_count)

@app.route('/orders')
def orders_page():
    orders = load_orders()
    cart   = load_cart()
    cart_count = sum(i['qty'] for i in cart)
    return render_template('orders.html', orders=orders, cart_count=cart_count)

@app.context_processor
def inject_globals():
    return {'year': datetime.now().year}

if __name__ == '__main__':
    app.run(debug=True, port=5000)