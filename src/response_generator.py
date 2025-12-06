"""
Response Generator V2 - Enhanced Template-based Response Generator

Key Improvements:
1. Support for 6 intents (including specific product intents)
2. WhatsApp fallback redirect for out-of-scope queries
3. More natural Indonesian language variations
4. Confidence-based fallback messages

Author: Firas
Version: 2.0
"""

import random
from typing import Dict, Optional


class ResponseGeneratorV2:
    """
    Template-based response generator with WhatsApp fallback
    """

    def __init__(self, whatsapp_number: str = "+6281234567890"):
        """
        Initialize response generator

        Args:
            whatsapp_number: WhatsApp number for human support redirect
        """
        self.whatsapp_number = whatsapp_number
        self._init_templates()

    def _init_templates(self):
        """Initialize response templates"""

        self.templates = {
            # ============================================================
            # ORDER STATUS
            # ============================================================
            'order_status': {
                'found': [
                    "Pesanan #{order_id} Anda saat ini berstatus **{status}**. {additional_info}",
                    "Status orderan #{order_id}: **{status}**. {additional_info}",
                    "Orderan #{order_id} sedang **{status}** nih. {additional_info}",
                    "Kabar baik! Pesanan #{order_id} statusnya **{status}**. {additional_info}",
                ],
                'not_found': [
                    "Maaf, pesanan dengan nomor #{order_id} tidak ditemukan. Pastikan nomor ordernya benar ya!",
                    "Hmm, pesanan #{order_id} tidak ada di sistem kami. Coba cek lagi nomor ordernya?",
                    "Pesanan #{order_id} tidak ditemukan. Mungkin salah ketik?",
                ],
                'need_order_id': [
                    "Untuk cek status pesanan, bisa kasih nomor ordernya? Contoh: #12345",
                    "Nomor order-nya berapa ya? Nanti saya bantuin cek statusnya.",
                    "Kasih tau nomor ordernya dong, biar bisa saya cek statusnya.",
                ]
            },

            # ============================================================
            # PAYMENT INFO
            # ============================================================
            'payment_info': {
                'list': [
                    "Kami menerima pembayaran melalui:\n{methods}\n\nPilih yang paling nyaman untuk Anda!",
                    "Metode pembayaran yang tersedia:\n{methods}\n\nSemua aman dan terpercaya!",
                    "Anda bisa bayar pakai:\n{methods}\n\nTinggal pilih sesuai kebutuhan!",
                ],
                'specific_method': [
                    "Ya, kami menerima pembayaran via **{method}**!",
                    "Tentu bisa bayar pakai **{method}**!",
                    "**{method}** tersedia sebagai metode pembayaran kami.",
                ],
                'not_available': [
                    "Maaf, informasi pembayaran tidak tersedia saat ini.",
                ]
            },

            # ============================================================
            # PRODUCT PRICE
            # ============================================================
            'product_price': {
                'found': [
                    "Harga **{product_name}** adalah **Rp {price:,}**",
                    "**{product_name}** dijual seharga **Rp {price:,}**",
                    "Untuk **{product_name}**, harganya **Rp {price:,}**",
                    "**{product_name}**: Rp {price:,}",
                ],
                'not_found': [
                    "Maaf, saya tidak menemukan produk yang Anda maksud. Coba kata kunci lain?",
                    "Produk tidak ditemukan. Bisa jelaskan lebih detail produk yang dicari?",
                ],
                'multiple': [
                    "Berikut harga produk yang cocok:\n{product_list}\n\nMau yang mana?",
                    "Ada beberapa produk yang sesuai:\n{product_list}",
                ]
            },

            # ============================================================
            # PRODUCT STOCK
            # ============================================================
            'product_stock': {
                'in_stock': [
                    "Stok **{product_name}**: **{stock} unit** tersedia",
                    "**{product_name}** masih ada **{stock} unit** ya!",
                    "Kabar baik! **{product_name}** ready stock **{stock} unit**",
                    "Tersedia **{stock} unit** untuk **{product_name}**",
                ],
                'out_of_stock': [
                    "Maaf, **{product_name}** sedang habis.",
                    "**{product_name}** stoknya kosong untuk saat ini.",
                    "Sayang sekali, **{product_name}** lagi out of stock.",
                ],
                'low_stock': [
                    "Stok **{product_name}** tinggal **{stock} unit** lagi! Buruan sebelum kehabisan!",
                    "**{product_name}** hampir habis, sisa **{stock} unit**!",
                ],
                'not_found': [
                    "Maaf, saya tidak menemukan produk yang Anda maksud untuk cek stok.",
                    "Produk tidak ditemukan. Bisa sebutkan nama produknya lebih jelas?",
                ],
                'featured_list': [
                    "Berikut produk yang tersedia di toko kami:\n{product_list}\n\nMau tanya detail produk mana?",
                    "Produk yang tersedia:\n{product_list}\n\nAda yang mau dicek lebih lanjut?",
                    "Ini daftar produk kami:\n{product_list}",
                ]
            },

            # ============================================================
            # PRODUCT DESCRIPTION
            # ============================================================
            'product_description': {
                'found': [
                    "**{product_name}**\n\n{description}\n\nHarga: Rp {price:,}",
                    "Ini info lengkap **{product_name}**:\n\n{description}\n\nHarga: Rp {price:,}",
                    "**{product_name}**\n{description}\n\nDijual seharga Rp {price:,}",
                ],
                'not_found': [
                    "Maaf, saya tidak menemukan produk yang Anda maksud.",
                    "Produk tidak ditemukan. Coba kata kunci yang berbeda?",
                ],
                'featured_list': [
                    "Berikut produk unggulan di toko kami:\n{product_list}\n\nProduk mana yang ingin diketahui detailnya?",
                    "Produk yang tersedia:\n{product_list}\n\nMau info detail yang mana?",
                    "Daftar produk kami:\n{product_list}",
                ]
            },

            # ============================================================
            # PRODUCT LIST (all available products)
            # ============================================================
            'product_list': {
                'list': [
                    "Berikut produk yang tersedia di toko kami:\n{product_list}\n\nMau tanya detail produk mana?",
                    "Produk yang tersedia:\n{product_list}\n\nAda yang menarik?",
                    "Ini daftar produk kami:\n{product_list}",
                ],
                'empty': [
                    "Maaf, belum ada produk yang tersedia saat ini.",
                    "Produk belum tersedia. Silakan cek lagi nanti!",
                ]
            },

            # ============================================================
            # SEARCH PRODUCT (keyword search)
            # ============================================================
            'search_product': {
                'found': [
                    "Hasil pencarian untuk **{keyword}**:\n\n**{product_name}**\nHarga: Rp {price:,}\n{description}",
                    "Ditemukan **{product_name}**!\n\nHarga: Rp {price:,}",
                    "**{product_name}** cocok dengan pencarian Anda.\nHarga: Rp {price:,}",
                ],
                'not_found': [
                    "Maaf, tidak ada produk yang cocok dengan **{keyword}**. Coba kata kunci lain?",
                    "Produk **{keyword}** tidak ditemukan. Mungkin coba variasi lain?",
                ],
                'suggestions': [
                    "Produk **{keyword}** tidak ditemukan. Berikut rekomendasi produk lain:\n{product_list}",
                ]
            },

            # ============================================================
            # PRODUCT BY CATEGORY
            # ============================================================
            'product_by_category': {
                'list': [
                    "Produk dalam kategori **{category}**:\n{product_list}\n\nMau lihat detail yang mana?",
                    "Berikut produk **{category}** kami:\n{product_list}",
                    "Kategori **{category}**:\n{product_list}",
                ],
                'empty': [
                    "Maaf, belum ada produk dalam kategori **{category}**.",
                    "Kategori **{category}** kosong saat ini.",
                ]
            },

            # ============================================================
            # PRODUCT BY ATTRIBUTE (brand, color, size)
            # ============================================================
            'product_by_attribute': {
                'found': [
                    "Ditemukan **{product_name}**!\n\nHarga: Rp {price:,}\nStok: {stock} unit",
                    "**{product_name}** tersedia!\nHarga: Rp {price:,}\nStok: {stock} unit",
                ],
                'not_found': [
                    "Maaf, produk dengan kriteria tersebut tidak ditemukan.",
                    "Tidak ada produk yang sesuai dengan atribut yang diminta.",
                ],
                'with_attributes': [
                    "Ditemukan **{product_name}** ({attributes})!\n\nHarga: Rp {price:,}\nStok: {stock} unit",
                ]
            },

            # ============================================================
            # OUT OF SCOPE / FALLBACK
            # ============================================================
            'out_of_scope': {
                'redirect': [
                    "Pertanyaan ini di luar kemampuan saya. Untuk bantuan lebih lanjut, silakan hubungi CS kami via WhatsApp: {whatsapp}",
                    "Maaf, saya belum bisa membantu dengan pertanyaan ini. Hubungi tim support kami di WhatsApp: {whatsapp}",
                    "Hmm, ini di luar area saya. Silakan chat CS kami langsung di: {whatsapp}",
                ],
                'low_confidence': [
                    "Saya kurang yakin dengan pertanyaan ini. Untuk info lebih akurat, hubungi CS kami: {whatsapp}",
                    "Maaf, saya tidak sepenuhnya paham maksud Anda. Bisa coba tanya CS kami di: {whatsapp}",
                    "Pertanyaan ini agak kompleks. Lebih baik langsung chat dengan tim kami: {whatsapp}",
                ],
            }
        }

        # Order status additional info
        self.status_info = {
            'pending': 'Pesanan sedang diproses oleh tim kami.',
            'pending_payment': 'Menunggu pembayaran. Segera lakukan pembayaran ya!',
            'processing': 'Pesanan sedang disiapkan untuk pengiriman.',
            'shipped': 'Pesanan sudah dikirim! Segera sampai.',
            'delivered': 'Pesanan sudah sampai. Terima kasih sudah berbelanja!',
            'canceled': 'Pesanan telah dibatalkan.',
            'completed': 'Pesanan sudah selesai. Terima kasih!',
            'closed': 'Pesanan sudah ditutup.',
        }

    def generate(
        self,
        intent: str,
        db_result: Dict,
        user_query: str = "",
        entities: Optional[Dict] = None
    ) -> str:
        """
        Generate response based on intent and database result

        Args:
            intent: Classified intent
            db_result: Database query result
            user_query: Original user query
            entities: Extracted entities

        Returns:
            Response string
        """
        entities = entities or {}

        if intent == 'order_status':
            return self._generate_order_response(db_result)

        elif intent == 'payment_info':
            return self._generate_payment_response(db_result, entities)

        elif intent == 'product_price':
            return self._generate_price_response(db_result)

        elif intent == 'product_stock':
            return self._generate_stock_response(db_result)

        elif intent == 'product_description':
            return self._generate_description_response(db_result)

        elif intent == 'product_list':
            return self._generate_product_list_response(db_result)

        elif intent == 'search_product':
            return self._generate_search_response(db_result, entities)

        elif intent == 'product_by_category':
            return self._generate_category_response(db_result, entities)

        elif intent == 'product_by_attribute':
            return self._generate_attribute_response(db_result, entities)

        elif intent == 'out_of_scope':
            return self.generate_fallback("out_of_scope", user_query)

        else:
            return self.generate_fallback("unknown", user_query)

    def generate_fallback(self, reason: str, original_query: str = "") -> str:
        """
        Generate fallback response with WhatsApp redirect

        Args:
            reason: Fallback reason ("out_of_scope", "low_confidence", "unknown")
            original_query: Original user query

        Returns:
            Fallback response with WhatsApp link
        """
        if reason == "out_of_scope":
            template = random.choice(self.templates['out_of_scope']['redirect'])
        else:
            template = random.choice(self.templates['out_of_scope']['low_confidence'])

        return template.format(whatsapp=self.whatsapp_number)

    def _generate_order_response(self, db_result: Dict) -> str:
        """Generate response for order status queries"""

        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"

        # Check if order data exists (LLM chatbot format: {"order": order_info})
        order = db_result.get('order')
        if order:
            template = random.choice(self.templates['order_status']['found'])
            status = order.get('status', 'unknown')
            additional_info = self.status_info.get(status, '')
            return template.format(
                order_id=order.get('id', 'N/A'),
                status=status.replace('_', ' ').title(),
                additional_info=additional_info
            )

        # Legacy format check (for backward compatibility)
        if db_result.get('found'):
            order = db_result.get('order', {})
            template = random.choice(self.templates['order_status']['found'])
            status = order.get('status', 'unknown')
            additional_info = self.status_info.get(status, '')
            return template.format(
                order_id=order.get('id', 'N/A'),
                status=status.replace('_', ' ').title(),
                additional_info=additional_info
            )

        # Order not found
        if db_result.get('order_id'):
            template = random.choice(self.templates['order_status']['not_found'])
            return template.format(order_id=db_result['order_id'])
        else:
            return random.choice(self.templates['order_status']['need_order_id'])

    def _generate_payment_response(self, db_result: Dict, entities: Dict) -> str:
        """Generate response for payment info queries"""

        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"

        if not db_result.get('found'):
            return random.choice(self.templates['payment_info']['not_available'])

        # Check if asking about specific method
        if entities.get('payment_method'):
            template = random.choice(self.templates['payment_info']['specific_method'])
            return template.format(method=entities['payment_method'].upper())

        # List all methods
        methods = db_result.get('methods', [])
        method_list = []

        for m in methods:
            title = m.get('title', 'Unknown')
            desc = m.get('description', '')
            method_list.append(f"- **{title}**: {desc}")

        methods_str = "\n".join(method_list)

        template = random.choice(self.templates['payment_info']['list'])
        return template.format(methods=methods_str)

    def _generate_price_response(self, db_result: Dict) -> str:
        """Generate response for product price queries"""

        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"

        # Handle single product format (from LLM chatbot)
        product = db_result.get('product')
        if product:
            template = random.choice(self.templates['product_price']['found'])
            return template.format(
                product_name=product.get('name', 'Produk'),
                price=int(product.get('price', 0))
            )

        # Legacy format with 'found' flag and 'products' list
        if not db_result.get('found'):
            return random.choice(self.templates['product_price']['not_found'])

        products = db_result.get('products', [])

        if len(products) == 1:
            p = products[0]
            template = random.choice(self.templates['product_price']['found'])
            return template.format(
                product_name=p.get('name', 'Produk'),
                price=int(p.get('price', 0))
            )

        # Multiple products
        product_list = []
        for p in products[:5]:  # Limit to 5
            product_list.append(
                f"- **{p.get('name')}**: Rp {int(p.get('price', 0)):,}"
            )

        template = random.choice(self.templates['product_price']['multiple'])
        return template.format(product_list="\n".join(product_list))

    def _generate_stock_response(self, db_result: Dict) -> str:
        """Generate response for product stock queries"""

        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"

        # Handle featured products list (no specific product requested)
        if db_result.get('type') == 'featured':
            products = db_result.get('products', [])
            if not products:
                return "Maaf, belum ada produk yang tersedia saat ini."

            product_list = []
            for p in products[:10]:  # Limit to 10
                name = p.get('name', 'Produk')
                price = int(p.get('price', 0))
                stock = p.get('stock', 0)
                stock_info = f"(stok: {stock})" if stock else "(habis)"
                product_list.append(f"- **{name}** - Rp {price:,} {stock_info}")

            template = random.choice(self.templates['product_stock']['featured_list'])
            return template.format(product_list="\n".join(product_list))

        # Handle single product query
        if db_result.get('product'):
            p = db_result['product']
            stock_info = p.get('stock', {})
            stock = int(stock_info.get('qty', 0)) if stock_info else 0
            name = p.get('name', 'Produk')

            if stock == 0:
                template = random.choice(self.templates['product_stock']['out_of_stock'])
                return template.format(product_name=name)
            elif stock <= 5:
                template = random.choice(self.templates['product_stock']['low_stock'])
                return template.format(product_name=name, stock=stock)
            else:
                template = random.choice(self.templates['product_stock']['in_stock'])
                return template.format(product_name=name, stock=stock)

        # Legacy format with 'found' flag
        if not db_result.get('found'):
            return random.choice(self.templates['product_stock']['not_found'])

        products = db_result.get('products', [])

        if len(products) == 0:
            return random.choice(self.templates['product_stock']['not_found'])

        p = products[0]
        stock = int(p.get('stock', 0))
        name = p.get('name', 'Produk')

        if stock == 0:
            template = random.choice(self.templates['product_stock']['out_of_stock'])
            return template.format(product_name=name)
        elif stock <= 5:
            template = random.choice(self.templates['product_stock']['low_stock'])
            return template.format(product_name=name, stock=stock)
        else:
            template = random.choice(self.templates['product_stock']['in_stock'])
            return template.format(product_name=name, stock=stock)

    def _generate_description_response(self, db_result: Dict) -> str:
        """Generate response for product description queries"""

        if db_result.get('error'):
            return f"Maaf, terjadi kesalahan: {db_result['error']}"

        # Handle featured products list (no specific product requested)
        if db_result.get('type') == 'featured':
            products = db_result.get('products', [])
            if not products:
                return "Maaf, belum ada produk yang tersedia saat ini."

            product_list = []
            for p in products[:10]:  # Limit to 10
                name = p.get('name', 'Produk')
                price = int(p.get('price', 0))
                product_list.append(f"- **{name}** - Rp {price:,}")

            template = random.choice(self.templates['product_description']['featured_list'])
            return template.format(product_list="\n".join(product_list))

        # Handle single product query
        if db_result.get('product'):
            p = db_result['product']
            template = random.choice(self.templates['product_description']['found'])
            return template.format(
                product_name=p.get('name', 'Produk'),
                description=p.get('description', 'Tidak ada deskripsi'),
                price=int(p.get('price', 0))
            )

        # Legacy format with 'found' flag
        if not db_result.get('found'):
            return random.choice(self.templates['product_description']['not_found'])

        products = db_result.get('products', [])

        if len(products) == 0:
            return random.choice(self.templates['product_description']['not_found'])

        p = products[0]
        template = random.choice(self.templates['product_description']['found'])

        return template.format(
            product_name=p.get('name', 'Produk'),
            description=p.get('description', 'Tidak ada deskripsi'),
            price=int(p.get('price', 0))
        )

    def _generate_product_list_response(self, db_result: Dict) -> str:
        """Generate response for product list queries (product_list task)."""

        if db_result.get('error'):
            return random.choice(self.templates['product_list']['empty'])

        products = db_result.get('products', [])
        if not products:
            return random.choice(self.templates['product_list']['empty'])

        product_list = []
        for p in products[:10]:  # Limit to 10
            name = p.get('name', 'Produk')
            price = int(p.get('price', 0))
            product_list.append(f"- **{name}** - Rp {price:,}")

        template = random.choice(self.templates['product_list']['list'])
        return template.format(product_list="\n".join(product_list))

    def _generate_search_response(self, db_result: Dict, entities: Dict) -> str:
        """Generate response for search product queries (search_product task)."""

        product_name = entities.get('product_name', 'produk')

        if db_result.get('error'):
            # Check for suggestions
            suggestions = db_result.get('suggestions', [])
            if suggestions:
                product_list = []
                for p in suggestions[:5]:
                    name = p.get('name', 'Produk')
                    price = int(p.get('price', 0))
                    product_list.append(f"- **{name}** - Rp {price:,}")
                template = random.choice(self.templates['search_product']['suggestions'])
                return template.format(
                    keyword=product_name,
                    product_list="\n".join(product_list)
                )
            template = random.choice(self.templates['search_product']['not_found'])
            return template.format(keyword=product_name)

        # Product found
        if db_result.get('product'):
            p = db_result['product']
            template = random.choice(self.templates['search_product']['found'])
            return template.format(
                keyword=product_name,
                product_name=p.get('name', 'Produk'),
                price=int(p.get('price', 0)),
                description=p.get('description', '')[:100] + '...' if len(p.get('description', '')) > 100 else p.get('description', '')
            )

        return random.choice(self.templates['search_product']['not_found']).format(keyword=product_name)

    def _generate_category_response(self, db_result: Dict, entities: Dict) -> str:
        """Generate response for category queries (product_by_category task)."""

        category = entities.get('category', 'kategori')

        if db_result.get('error'):
            template = random.choice(self.templates['product_by_category']['empty'])
            return template.format(category=category)

        products = db_result.get('products', [])
        if not products:
            template = random.choice(self.templates['product_by_category']['empty'])
            return template.format(category=category)

        product_list = []
        for p in products[:10]:
            name = p.get('name', 'Produk')
            price = int(p.get('price', 0))
            product_list.append(f"- **{name}** - Rp {price:,}")

        template = random.choice(self.templates['product_by_category']['list'])
        return template.format(
            category=category,
            product_list="\n".join(product_list)
        )

    def _generate_attribute_response(self, db_result: Dict, entities: Dict) -> str:
        """Generate response for attribute queries (product_by_attribute task)."""

        if db_result.get('error'):
            return random.choice(self.templates['product_by_attribute']['not_found'])

        if db_result.get('product'):
            p = db_result['product']
            stock_info = p.get('stock', {})
            stock = int(stock_info.get('qty', 0)) if stock_info else 0

            # Build attributes string
            attrs = entities.get('attributes', {})
            attr_parts = []
            if attrs.get('brand'):
                attr_parts.append(attrs['brand'])
            if attrs.get('color'):
                attr_parts.append(attrs['color'])
            if attrs.get('size'):
                attr_parts.append(f"size {attrs['size']}")
            attributes_str = ", ".join(attr_parts) if attr_parts else ""

            if attributes_str:
                template = random.choice(self.templates['product_by_attribute']['with_attributes'])
                return template.format(
                    product_name=p.get('name', 'Produk'),
                    attributes=attributes_str,
                    price=int(p.get('price', 0)),
                    stock=stock
                )
            else:
                template = random.choice(self.templates['product_by_attribute']['found'])
                return template.format(
                    product_name=p.get('name', 'Produk'),
                    price=int(p.get('price', 0)),
                    stock=stock
                )

        return random.choice(self.templates['product_by_attribute']['not_found'])


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESPONSE GENERATOR V2 TEST")
    print("=" * 60)

    generator = ResponseGeneratorV2(whatsapp_number="+6281234567890")

    # Test order status
    print("\n1. Order Status (Found):")
    result = generator.generate(
        intent='order_status',
        db_result={
            'found': True,
            'order': {'id': '12345', 'status': 'shipped'}
        }
    )
    print(f"   {result}")

    # Test product price
    print("\n2. Product Price:")
    result = generator.generate(
        intent='product_price',
        db_result={
            'found': True,
            'products': [{'name': 'Arctic Beanie', 'price': 150000}]
        }
    )
    print(f"   {result}")

    # Test product stock
    print("\n3. Product Stock (Low):")
    result = generator.generate(
        intent='product_stock',
        db_result={
            'found': True,
            'products': [{'name': 'Winter Scarf', 'stock': 3}]
        }
    )
    print(f"   {result}")

    # Test fallback
    print("\n4. Out of Scope Fallback:")
    result = generator.generate_fallback("out_of_scope", "siapa presiden indonesia?")
    print(f"   {result}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
