"""
Template-based Response Generator
"""

import random

class TemplateResponseGenerator:
    
    def __init__(self):
        self.templates = {
            'order_status': {
                'found': [
                    "Pesanan #{order_id} Anda saat ini berstatus **{status}**. {additional_info}",
                    "Status orderan #{order_id}: **{status}**. {additional_info}",
                    "Orderan #{order_id} sedang **{status}** nih. {additional_info}",
                ],
                'not_found': [
                    "Maaf, saya tidak menemukan pesanan dengan nomor {order_id}. Pastikan nomor ordernya benar ya!",
                    "Pesanan {order_id} tidak ditemukan di sistem. Coba cek lagi nomor ordernya?",
                ],
                'need_order_id': [
                    "Untuk cek status pesanan, bisa kasih nomor ordernya? Contoh: #12345",
                    "Nomor order-nya berapa ya? Nanti saya bantuin cek statusnya.",
                ]
            },
            'payment_info': {
                'list': [
                    "Kami menerima pembayaran melalui: {methods}. Pilih yang paling nyaman untuk Anda!",
                    "Metode pembayaran yang tersedia: {methods}. Semua aman dan terpercaya!",
                    "Anda bisa bayar pakai: {methods}. Tinggal pilih sesuai kebutuhan!",
                ],
            },
            'product_info': {
                'single': [
                    "**{product_name}**\n💰 Harga: Rp {price:,}\n📦 Stok: {stock} unit\n📝 {description}",
                ],
                'not_found': [
                    "Maaf, produk '{query}' tidak ditemukan. Coba kata kunci lain atau cek katalog kami!",
                    "Saya tidak menemukan produk '{query}'. Mungkin maksudnya produk lain?",
                ],
                'multiple': "Saya menemukan {count} produk:\n\n{product_list}\n\nPilih yang mana?"
            }
        }
        
        self.status_info = {
            'pending': 'Pesanan sedang diproses oleh tim kami.',
            'pending_payment': 'Menunggu pembayaran. Segera lakukan pembayaran ya!',
            'processing': 'Pesanan sedang disiapkan untuk pengiriman.',
            'shipped': 'Pesanan sudah dikirim! Segera sampai.',
            'delivered': 'Pesanan sudah sampai. Terima kasih sudah berbelanja!',
            'canceled': 'Pesanan telah dibatalkan.',
        }
    
    def generate(self, intent, db_result, user_query=""):
        """Main function untuk generate response"""
        
        if intent == 'order_status':
            return self._generate_order_status_response(db_result)
        
        elif intent == 'payment_info':
            return self._generate_payment_response(db_result)
        
        elif intent == 'product_info':
            return self._generate_product_response(db_result, user_query)
        
        else:
            return "Maaf, saya belum bisa membantu dengan pertanyaan ini."
    
    def _generate_order_status_response(self, db_result):
        """Generate response untuk order status"""
        
        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"
        
        if not db_result.get('found'):
            if db_result.get('order_id'):
                template = random.choice(self.templates['order_status']['not_found'])
                return template.format(order_id=db_result['order_id'])
            else:
                return random.choice(self.templates['order_status']['need_order_id'])
        
        # Order found
        order = db_result['order']
        template = random.choice(self.templates['order_status']['found'])
        status = order['status']
        additional_info = self.status_info.get(status, '')
        
        return template.format(
            order_id=order['id'],
            status=status.replace('_', ' ').title(),
            additional_info=additional_info
        )
    
    def _generate_payment_response(self, db_result):
        """Generate response untuk payment info"""
        
        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"
        
        if not db_result.get('found'):
            return "Maaf, informasi pembayaran tidak tersedia saat ini."
        
        methods = db_result['methods']
        method_list = []
        
        for m in methods:
            method_list.append(f"• **{m['title']}**: {m['description']}")
        
        methods_str = "\n".join(method_list)
        
        template = random.choice(self.templates['payment_info']['list'])
        return template.format(methods=methods_str)
    
    def _generate_product_response(self, db_result, user_query):
        """Generate response untuk product info"""
        
        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"
        
        if not db_result.get('found'):
            template = random.choice(self.templates['product_info']['not_found'])
            return template.format(query=db_result.get('query', user_query))
        
        products = db_result['products']
        
        if len(products) == 1:
            # Single product
            p = products[0]
            template = random.choice(self.templates['product_info']['single'])
            
            return template.format(
                product_name=p['name'],
                price=p['price'],
                stock=p['stock'],
                description=p['description']
            )
        
        else:
            # Multiple products
            product_list = []
            for p in products:
                product_list.append(f"• **{p['name']}** - Rp {p['price']:,} (Stok: {p['stock']})")
            
            product_str = "\n".join(product_list)
            
            return self.templates['product_info']['multiple'].format(
                count=len(products),
                product_list=product_str
            )