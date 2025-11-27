"""
Bagisto Query Handler - Production Ready
Matched dengan actual Bagisto database schema
Fixed: MySQL strict mode GROUP BY issues
"""

import mysql.connector
from mysql.connector import Error
import re
from datetime import datetime

class BagistoQueryHandler:
    
    def __init__(self, db_config=None):
        """
        Initialize with Bagisto database connection
        
        Args:
            db_config (dict): Database configuration
        """
        
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'bagisto_db',
                'port': 3306
            }
        
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        
        # Status mapping untuk user-friendly messages
        self.status_mapping = {
            'pending': 'Menunggu Pembayaran',
            'processing': 'Sedang Diproses',
            'completed': 'Selesai',
            'canceled': 'Dibatalkan',
            'closed': 'Ditutup',
            'pending_payment': 'Menunggu Pembayaran',
            'fraud': 'Terdeteksi Fraud',
        }
        
        # Connect to database
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                
                # Disable ONLY_FULL_GROUP_BY mode to prevent GROUP BY issues
                try:
                    self.cursor.execute("SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''))")
                except:
                    pass  # Ignore if fails
                
                db_info = self.connection.get_server_info()
                print(f"✅ Connected to Bagisto MySQL Server v{db_info}")
                return True
        
        except Error as e:
            print(f"❌ Database connection error: {e}")
            self.connection = None
            self.cursor = None
            return False
    
    def _reconnect(self):
        """Reconnect if connection lost"""
        if self.connection is None or not self.connection.is_connected():
            print("🔄 Reconnecting to database...")
            return self._connect()
        return True
    
    def _execute_query(self, query, params=None):
        """
        Execute SQL query with auto-reconnect
        
        Args:
            query (str): SQL query
            params (tuple): Query parameters
        
        Returns:
            list: Query results or None
        """
        try:
            # Ensure connection
            if not self._reconnect():
                return None
            
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        
        except Error as e:
            print(f"❌ Query error: {e}")
            print(f"   Query: {query}")
            print(f"   Params: {params}")
            return None
    
    def _clean_id(self, id_value):
        """Clean ID (remove #, spaces, etc)"""
        if id_value:
            return str(id_value).replace('#', '').replace(' ', '').strip()
        return None
    
    # ========================================================
    # ORDER STATUS QUERIES
    # ========================================================
    
    def get_order_status(self, order_id=None):
        """
        Query order status from Bagisto
        
        Bagisto uses:
        - orders.id or orders.increment_id as order identifier
        - orders.status (pending, processing, completed, etc)
        
        Args:
            order_id (str): Order ID or increment ID
        
        Returns:
            dict: Order information
        """
        
        try:
            if not order_id:
                return {
                    'found': False,
                    'message': 'Nomor order tidak disebutkan. Contoh: order #1 atau order 1'
                }
            
            order_id = self._clean_id(order_id)
            
            # Query order - search by both id and increment_id
            query = """
                SELECT 
                    id,
                    increment_id,
                    status,
                    customer_first_name,
                    customer_last_name,
                    customer_email,
                    grand_total,
                    order_currency_code,
                    total_item_count,
                    total_qty_ordered,
                    shipping_method,
                    shipping_title,
                    created_at,
                    updated_at
                FROM orders
                WHERE id = %s OR increment_id = %s
                LIMIT 1
            """
            
            result = self._execute_query(query, (order_id, order_id))
            
            if result and len(result) > 0:
                order = result[0]
                
                # Format status untuk user
                status_raw = order['status']
                status_display = self.status_mapping.get(status_raw, status_raw.title())
                
                # Format date
                created_date = order['created_at']
                if isinstance(created_date, datetime):
                    created_date = created_date.strftime('%d %B %Y')
                
                return {
                    'found': True,
                    'order': {
                        'id': str(order['increment_id']),  # Use increment_id for display
                        'status': status_raw,
                        'status_display': status_display,
                        'customer_name': f"{order['customer_first_name']} {order['customer_last_name']}",
                        'customer_email': order['customer_email'],
                        'total': float(order['grand_total']),
                        'currency': order['order_currency_code'],
                        'item_count': int(order['total_item_count']) if order['total_item_count'] else 0,
                        'qty_ordered': int(order['total_qty_ordered']) if order['total_qty_ordered'] else 0,
                        'shipping_method': order['shipping_title'] or order['shipping_method'],
                        'date': created_date
                    }
                }
            else:
                return {
                    'found': False,
                    'order_id': order_id,
                    'message': f'Order dengan nomor {order_id} tidak ditemukan di sistem'
                }
        
        except Exception as e:
            print(f"❌ Error in get_order_status: {e}")
            return {
                'error': 'Terjadi kesalahan saat mengecek order',
                'details': str(e)
            }
    
    def get_order_items(self, order_id):
        """
        Get order items/products
        
        Args:
            order_id (str): Order ID
        
        Returns:
            dict: Order items information
        """
        try:
            order_id = self._clean_id(order_id)
            
            query = """
                SELECT 
                    oi.name as product_name,
                    oi.sku,
                    oi.qty_ordered,
                    oi.price,
                    oi.total,
                    o.increment_id as order_id
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                WHERE o.id = %s OR o.increment_id = %s
            """
            
            result = self._execute_query(query, (order_id, order_id))
            
            if result and len(result) > 0:
                items = []
                for item in result:
                    items.append({
                        'name': item['product_name'],
                        'sku': item['sku'],
                        'qty': int(item['qty_ordered']),
                        'price': float(item['price']),
                        'total': float(item['total'])
                    })
                
                return {
                    'found': True,
                    'order_id': result[0]['order_id'],
                    'items': items,
                    'item_count': len(items)
                }
            else:
                return {
                    'found': False,
                    'message': 'Item order tidak ditemukan'
                }
        
        except Exception as e:
            print(f"❌ Error in get_order_items: {e}")
            return {'error': str(e)}
    
    # ========================================================
    # PAYMENT METHOD QUERIES
    # ========================================================
    
    def get_payment_methods(self):
        """
        Get available payment methods
        
        Bagisto stores payment config in core_config table
        with keys like 'sales.paymentmethods.cashondelivery.active'
        
        Returns:
            dict: Available payment methods
        """
        
        try:
            # Query payment config from core_config
            query = """
                SELECT code, value
                FROM core_config
                WHERE code LIKE 'sales.paymentmethods.%.active'
                  AND value = '1'
            """
            
            result = self._execute_query(query)
            
            # Map Bagisto payment method codes to user-friendly names
            payment_methods_info = {
                'cashondelivery': {
                    'title': 'Cash on Delivery (COD)',
                    'description': 'Bayar saat barang sampai ke alamat Anda'
                },
                'moneytransfer': {
                    'title': 'Transfer Bank',
                    'description': 'Transfer ke rekening bank (BCA, Mandiri, BNI, BRI)'
                },
                'paypal_standard': {
                    'title': 'PayPal',
                    'description': 'Pembayaran online via PayPal'
                },
                'paypal_smart_button': {
                    'title': 'PayPal Smart Button',
                    'description': 'Pembayaran cepat dengan PayPal'
                },
            }
            
            methods = []
            
            if result and len(result) > 0:
                # Parse active payment methods from config
                for row in result:
                    # Extract method name from code
                    # e.g., 'sales.paymentmethods.cashondelivery.active' -> 'cashondelivery'
                    code_parts = row['code'].split('.')
                    if len(code_parts) >= 3:
                        method_code = code_parts[2]
                        
                        if method_code in payment_methods_info:
                            method_info = payment_methods_info[method_code]
                            methods.append({
                                'method': method_code,
                                'title': method_info['title'],
                                'description': method_info['description']
                            })
            
            # If no methods found in config, return default methods
            if len(methods) == 0:
                methods = [
                    {
                        'method': 'cashondelivery',
                        'title': 'Cash on Delivery (COD)',
                        'description': 'Bayar saat barang sampai ke alamat Anda'
                    },
                    {
                        'method': 'moneytransfer',
                        'title': 'Transfer Bank',
                        'description': 'Transfer ke rekening bank (BCA, Mandiri, BNI, BRI)'
                    },
                ]
            
            return {
                'found': True,
                'methods': methods
            }
        
        except Exception as e:
            print(f"❌ Error in get_payment_methods: {e}")
            # Return default methods as fallback
            return {
                'found': True,
                'methods': [
                    {
                        'method': 'cashondelivery',
                        'title': 'Cash on Delivery (COD)',
                        'description': 'Bayar saat barang sampai'
                    },
                    {
                        'method': 'moneytransfer',
                        'title': 'Transfer Bank',
                        'description': 'Transfer ke rekening bank'
                    },
                ]
            }
    
    # ========================================================
    # PRODUCT QUERIES
    # ========================================================
    
    def get_product_info(self, product_name=None):
        """
        Query product information from Bagisto
        
        Bagisto structure:
        - products table: base product data (id, sku, type)
        - product_flat table: localized product data (name, price, description)
        - product_inventories table: stock quantity
        
        Args:
            product_name (str): Product name to search
        
        Returns:
            dict: Product information
        """
        
        try:
            if not product_name or len(product_name.strip()) == 0:
                # Return featured products
                return self._get_featured_products()
            
            product_name = product_name.lower().strip()
            
            # Search products by name or SKU
            # Using subquery for stock to avoid GROUP BY issues
            query = """
                SELECT 
                    pf.product_id as id,
                    pf.sku,
                    pf.name,
                    pf.price,
                    pf.short_description,
                    pf.url_key,
                    (
                        SELECT COALESCE(SUM(pi.qty), 0) 
                        FROM product_inventories pi 
                        WHERE pi.product_id = pf.product_id
                    ) as total_stock
                FROM product_flat pf
                WHERE (LOWER(pf.name) LIKE %s
                   OR LOWER(pf.sku) LIKE %s
                   OR LOWER(pf.short_description) LIKE %s)
                  AND pf.status = 1
                  AND pf.visible_individually = 1
                LIMIT 5
            """
            
            search_term = f"%{product_name}%"
            result = self._execute_query(query, (search_term, search_term, search_term))
            
            if result and len(result) > 0:
                products = []
                for row in result:
                    products.append({
                        'id': row['id'],
                        'sku': row['sku'],
                        'name': row['name'],
                        'price': float(row['price']) if row['price'] else 0,
                        'stock': int(row['total_stock']) if row['total_stock'] else 0,
                        'description': (row['short_description'] or 'Produk berkualitas')[:200],
                        'url_key': row['url_key']
                    })
                
                return {
                    'found': True,
                    'products': products,
                    'count': len(products)
                }
            else:
                return {
                    'found': False,
                    'query': product_name,
                    'message': f'Produk "{product_name}" tidak ditemukan'
                }
        
        except Exception as e:
            print(f"❌ Error in get_product_info: {e}")
            return {
                'error': 'Terjadi kesalahan saat mencari produk',
                'details': str(e)
            }
    
    def _get_featured_products(self, limit=5):
        """
        Get featured or newest products
        
        Args:
            limit (int): Number of products to return
        
        Returns:
            dict: Featured products
        """
        try:
            # Simple query without complex GROUP BY
            # Using subquery for stock calculation
            query = """
                SELECT 
                    pf.product_id as id,
                    pf.sku,
                    pf.name,
                    pf.price,
                    pf.short_description,
                    (
                        SELECT COALESCE(SUM(pi.qty), 0) 
                        FROM product_inventories pi 
                        WHERE pi.product_id = pf.product_id
                    ) as total_stock
                FROM product_flat pf
                WHERE pf.status = 1
                  AND pf.visible_individually = 1
                ORDER BY pf.created_at DESC
                LIMIT %s
            """
            
            result = self._execute_query(query, (limit,))
            
            if result and len(result) > 0:
                products = []
                for row in result:
                    products.append({
                        'id': row['id'],
                        'sku': row['sku'],
                        'name': row['name'],
                        'price': float(row['price']) if row['price'] else 0,
                        'stock': int(row['total_stock']) if row['total_stock'] else 0,
                        'description': (row['short_description'] or 'Produk berkualitas')[:200]
                    })
                
                return {
                    'found': True,
                    'products': products,
                    'count': len(products)
                }
            else:
                return {
                    'found': False,
                    'message': 'Tidak ada produk tersedia'
                }
        
        except Exception as e:
            print(f"❌ Error in _get_featured_products: {e}")
            return {'error': str(e)}
    
    def get_product_stock(self, product_id=None, sku=None):
        """
        Get specific product stock information
        
        Args:
            product_id (int): Product ID
            sku (str): Product SKU
        
        Returns:
            dict: Stock information
        """
        try:
            if not product_id and not sku:
                return {'error': 'Product ID or SKU required'}
            
            if sku:
                # Get product_id from SKU first
                query = "SELECT product_id FROM product_flat WHERE sku = %s LIMIT 1"
                result = self._execute_query(query, (sku,))
                
                if result and len(result) > 0:
                    product_id = result[0]['product_id']
                else:
                    return {'found': False, 'message': 'Product not found'}
            
            # Get total stock
            query = """
                SELECT 
                    pi.product_id,
                    SUM(pi.qty) as total_qty,
                    pf.name,
                    pf.sku
                FROM product_inventories pi
                JOIN product_flat pf ON pi.product_id = pf.product_id
                WHERE pi.product_id = %s
                GROUP BY pi.product_id, pf.name, pf.sku
            """
            
            result = self._execute_query(query, (product_id,))
            
            if result and len(result) > 0:
                stock_info = result[0]
                return {
                    'found': True,
                    'product_id': stock_info['product_id'],
                    'sku': stock_info['sku'],
                    'name': stock_info['name'],
                    'stock': int(stock_info['total_qty'])
                }
            else:
                return {
                    'found': False,
                    'message': 'Stock information not available'
                }
        
        except Exception as e:
            print(f"❌ Error in get_product_stock: {e}")
            return {'error': str(e)}
    
    # ========================================================
    # UTILITY METHODS
    # ========================================================
    
    def test_connection(self):
        """Test database connection and show info"""
        try:
            if self.connection and self.connection.is_connected():
                query = """
                    SELECT 
                        DATABASE() as db_name,
                        VERSION() as version,
                        (SELECT COUNT(*) FROM orders) as total_orders,
                        (SELECT COUNT(*) FROM product_flat) as total_products,
                        (SELECT COUNT(*) FROM customers) as total_customers
                """
                result = self._execute_query(query)
                
                if result and len(result) > 0:
                    info = result[0]
                    print("\n" + "="*60)
                    print("✅ DATABASE CONNECTION TEST")
                    print("="*60)
                    print(f"Database: {info['db_name']}")
                    print(f"MySQL Version: {info['version']}")
                    print(f"\n📊 Statistics:")
                    print(f"   Total Orders: {info['total_orders']}")
                    print(f"   Total Products: {info['total_products']}")
                    print(f"   Total Customers: {info['total_customers']}")
                    print("="*60)
                    return True
            else:
                print("❌ Not connected to database")
                return False
        
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection and self.connection.is_connected():
                self.connection.close()
                print("✅ Database connection closed")
        except Exception as e:
            print(f"Error closing connection: {e}")
    
    def __del__(self):
        """Destructor"""
        self.close()


# ============================================================
# TEST QUERY HANDLER
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("🧪 TESTING BAGISTO QUERY HANDLER")
    print("="*70)
    
    # Initialize handler
    handler = BagistoQueryHandler()
    
    # Test connection
    print("\n1️⃣  Testing connection...")
    handler.test_connection()
    
    # Test order status
    print("\n2️⃣  Testing order status query...")
    result = handler.get_order_status('1')
    if result.get('found'):
        order = result['order']
        print(f"✅ Order found!")
        print(f"   Order ID: {order['id']}")
        print(f"   Status: {order['status_display']}")
        print(f"   Customer: {order['customer_name']}")
        print(f"   Total: {order['currency']} {order['total']:,.0f}")
        print(f"   Date: {order['date']}")
    else:
        print(f"❌ {result.get('message', 'Order not found')}")
    
    # Test payment methods
    print("\n3️⃣  Testing payment methods...")
    result = handler.get_payment_methods()
    if result.get('found'):
        print(f"✅ Found {len(result['methods'])} payment methods:")
        for method in result['methods']:
            print(f"   • {method['title']}: {method['description']}")
    
    # Test product search
    print("\n4️⃣  Testing product search...")
    result = handler.get_product_info('beanie')
    if result.get('found'):
        print(f"✅ Found {result['count']} products:")
        for product in result['products']:
            print(f"\n   • {product['name']}")
            print(f"     SKU: {product['sku']}")
            print(f"     Price: IDR {product['price']:,.0f}")
            print(f"     Stock: {product['stock']} units")
    else:
        print(f"❌ {result.get('message', 'Products not found')}")
    
    # Test featured products
    print("\n5️⃣  Testing featured products (no search term)...")
    result = handler.get_product_info('')
    if result.get('found'):
        print(f"✅ Found {result['count']} featured products:")
        for product in result['products']:
            print(f"   • {product['name']} - IDR {product['price']:,.0f}")
    
    # Close connection
    print("\n6️⃣  Closing connection...")
    handler.close()
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)