from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, flash
import os
from datetime import datetime, date
import sqlite3
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import json
from fpdf import FPDF
import io
from functools import wraps

app = Flask(__name__)
app.secret_key = 'circuitfix_pro_secret_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load AI Model
def load_ai_model():
    try:
        # Try to load our trained model (will exist after training completes)
        model_path = 'runs/detect/circuitfix_ai_gpu3/weights/best.pt'
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print("✅ Custom PCB defect model loaded!")
            return model
        else:
            # Fallback for now - will switch to trained model automatically when ready
            model = YOLO('runs/detect/train/weights/best.pt')
            print("⚡ Training in progress - using fast model for now")
            return model
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return None

pcb_model = load_ai_model()

# Initialize Database
def init_db():
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT, 
                  email TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS analyses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  user_id INTEGER, 
                  image_path TEXT, 
                  defects TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS forum_posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  title TEXT,
                  content TEXT,
                  category TEXT DEFAULT 'General',
                  views INTEGER DEFAULT 0,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS forum_replies
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  post_id INTEGER,
                  user_id INTEGER,
                  content TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(post_id) REFERENCES forum_posts(id),
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.')
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

# Repair Knowledge Base
REPAIR_KNOWLEDGE = {
    'missing_hole': {
        'name': 'Missing Drill Hole',
        'severity': 'Medium',
        'steps': [
            'Use precision drill with 0.8-1.2mm bit',
            'Clean drill area with isopropyl alcohol', 
            'Drill at slow speed to prevent copper tearing',
            'Deburr hole edges',
            'Inspect for clean through-hole'
        ],
        'tools': ['Micro drill', 'Safety glasses', 'Drill bits', 'Isopropyl alcohol'],
        'time_estimate': '15-30 minutes',
        'cost_estimate': '$5-20'
    },
    'open_circuit': {
        'name': 'Open Circuit Trace',
        'severity': 'High', 
        'steps': [
            'Clean trace area with isopropyl alcohol',
            'Use fine-grit sandpaper to expose copper',
            'Apply flux to the trace',
            'Solder jumper wire across break',
            'Test continuity with multimeter'
        ],
        'tools': ['Soldering iron', 'Jumper wire', 'Flux', 'Multimeter'],
        'time_estimate': '20-45 minutes',
        'cost_estimate': '$10-30'
    },
    'short': {
        'name': 'Short Circuit',
        'severity': 'Critical',
        'steps': [
            'Identify short location with multimeter',
            'Use exacto knife to carefully separate traces',
            'Clean area with PCB cleaner',
            'Verify short is removed with continuity test'
        ],
        'tools': ['Multimeter', 'Exacto knife', 'PCB cleaner'],
        'time_estimate': '30-60 minutes', 
        'cost_estimate': '$15-50'
    },
    'mouse_bite': {
        'name': 'Mouse Bite Damage',
        'severity': 'Low',
        'steps': [
            'Use PCB cutter to trim rough edges',
            'Sand edges with 400-grit sandpaper',
            'Clean with compressed air',
            'Apply conformal coating if needed'
        ],
        'tools': ['PCB cutter', 'Sandpaper', 'Compressed air'],
        'time_estimate': '10-20 minutes',
        'cost_estimate': '$2-10'
    },
    'spur': {
        'name': 'Copper Spur',
        'severity': 'Low',
        'steps': [
            'Identify spur location under good lighting',
            'Use exacto knife to carefully remove protrusion',
            'Smooth area with fine sandpaper',
            'Clean with alcohol'
        ],
        'tools': ['Exacto knife', 'Fine sandpaper', 'Isopropyl alcohol'],
        'time_estimate': '10-15 minutes',
        'cost_estimate': '$2-8'
    },
    'spurious_copper': {
        'name': 'Spurious Copper',
        'severity': 'Medium',
        'steps': [
            'Apply UV solder mask over affected area',
            'Cure with UV light for 2-3 minutes',
            'Alternative: carefully scrape off excess copper',
            'Clean and inspect repair'
        ],
        'tools': ['UV solder mask', 'UV light', 'Scraper tool'],
        'time_estimate': '15-25 minutes',
        'cost_estimate': '$8-25'
    }
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_pcb_image(image_path):
    """Run REAL AI analysis on PCB image"""
    if pcb_model is None:
        return []
    
    try:
        # Run YOLO detection with confidence threshold
        results = pcb_model(image_path, conf=0.5, iou=0.45)
        
        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                defect_type = r.names[class_id]
                detections.append({
                    'defect': defect_type,
                    'confidence': round(confidence, 3),
                    'bbox': [int(x) for x in bbox],
                    'repair_advice': REPAIR_KNOWLEDGE.get(defect_type, {})
                })
        
        return detections
    except Exception as e:
        print(f"AI Analysis error: {e}")
        return []

def draw_bounding_boxes(image_path, detections):
    """Draw bounding boxes on image and save"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Draw each detection
        for detection in detections:
            bbox = detection['bbox']
            defect = detection['defect']
            confidence = detection['confidence']
            
            # Draw rectangle
            color = get_color_for_defect(defect)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            
            # Draw label background
            label = f"{defect} {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save annotated image
        annotated_path = image_path.replace('.jpg', '_annotated.jpg').replace('.png', '_annotated.png')
        cv2.imwrite(annotated_path, image)
        return annotated_path
    except Exception as e:
        print(f"Bounding box error: {e}")
        return None

def get_color_for_defect(defect_type):
    """Get color for different defect types"""
    colors = {
        'missing_hole': (0, 165, 255),      # Orange
        'open_circuit': (255, 0, 0),        # Red
        'short': (0, 0, 255),               # Blue
        'mouse_bite': (0, 255, 0),          # Green
        'spur': (255, 255, 0),              # Cyan
        'spurious_copper': (255, 0, 255)    # Magenta
    }
    return colors.get(defect_type, (128, 128, 128))  # Default gray

def create_pdf_report(analysis_data, username, image_path=None):
    """Generate PDF report with annotated image"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'CircuitFix - PCB Analysis Report', 0, 1, 'C')
    pdf.ln(5)
    
    # User info
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Technician: {username}', 0, 1)
    pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.ln(10)
    
    # Add annotated image if available
    if image_path and os.path.exists(image_path):
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Analyzed PCB Image:', 0, 1)
        pdf.ln(5)
        try:
            # Calculate image dimensions to fit page width (max 180mm width)
            pdf.image(image_path, x=15, w=180)
            pdf.ln(10)
        except Exception as e:
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, f'(Image could not be loaded: {str(e)})', 0, 1)
            pdf.ln(5)
    
    # Defects summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Defects Summary:', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    for i, detection in enumerate(analysis_data['detections'], 1):
        pdf.cell(0, 10, f'{i}. {detection["defect"]} - {detection["confidence"]*100:.1f}% confidence', 0, 1)
    
    pdf.ln(10)
    
    # Repair instructions
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Repair Instructions:', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    for detection in analysis_data['detections']:
        advice = detection['repair_advice']
        if advice:
            pdf.cell(0, 10, f'{advice["name"]} ({advice["severity"]} severity):', 0, 1)
            for step in advice.get('steps', []):
                pdf.cell(10)  # indent
                pdf.cell(0, 10, f'- {step}', 0, 1)
            pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin1')

@app.route('/')
def home():
    return render_template('index.html', 
                         logged_in='user_id' in session,
                         username=session.get('username'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', password)
        email = request.form['email']
        
        # Validate password confirmation
        if password != confirm_password:
            flash('Passwords do not match. Please try again.')
            return redirect('/register')
        
        # Validate password strength
        if len(password) < 8:
            flash('Password must be at least 8 characters long.')
            return redirect('/register')
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        conn = sqlite3.connect('circuitfix.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", 
                     (username, hashed_password, email))
            conn.commit()
            session['user_id'] = c.lastrowid
            session['username'] = username
            conn.close()
            flash('Account created successfully! Welcome to CircuitFix.')
            return redirect('/dashboard')
        except sqlite3.IntegrityError:
            conn.close()
            flash('Username already exists. Please choose a different username.')
            return redirect('/register')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('circuitfix.db')
        c = conn.cursor()
        c.execute("SELECT id, username, password FROM users WHERE username = ?", 
                 (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect('/dashboard')
        else:
            flash('Invalid credentials. Please check your username and password.')
            return redirect('/login')
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's analysis history
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    c.execute('''SELECT id, image_path, defects, created_at 
                 FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT 10''', 
              (session['user_id'],))
    history = c.fetchall()
    conn.close()
    
    return render_template('dashboard.html', 
                         username=session['username'],
                         history=history)

@app.route('/api/stats')
@login_required
def get_stats():
    """Get real statistics for the dashboard"""
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    
    # Today's analyses count
    today = date.today().isoformat()
    c.execute('''SELECT COUNT(*) FROM analyses 
                 WHERE user_id = ? AND DATE(created_at) = ?''',
              (session['user_id'], today))
    today_count = c.fetchone()[0]
    
    # Total analyses
    c.execute('SELECT COUNT(*) FROM analyses WHERE user_id = ?', 
              (session['user_id'],))
    total_analyses = c.fetchone()[0]
    
    # Total defects found
    c.execute('SELECT defects FROM analyses WHERE user_id = ?', 
              (session['user_id'],))
    total_defects = 0
    for row in c.fetchall():
        try:
            defects = json.loads(row[0])
            total_defects += len(defects)
        except:
            pass
    
    # Success rate (analyses with no critical defects)
    c.execute('SELECT defects FROM analyses WHERE user_id = ?', 
              (session['user_id'],))
    successful = 0
    for row in c.fetchall():
        try:
            defects = json.loads(row[0])
            has_critical = any(d.get('repair_advice', {}).get('severity') == 'Critical' for d in defects)
            if not has_critical:
                successful += 1
        except:
            pass
    
    success_rate = (successful / total_analyses * 100) if total_analyses > 0 else 100
    
    conn.close()
    
    return jsonify({
        'today_count': today_count,
        'total_analyses': total_analyses,
        'defects_found': total_defects,
        'success_rate': round(success_rate, 1),
        'avg_time': '2.3s'  # Average analysis time (could be tracked in DB)
    })

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(f"{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run AI analysis
        detections = analyze_pcb_image(filepath)
        
        # Draw bounding boxes on image
        annotated_image_path = draw_bounding_boxes(filepath, detections)
        
        # Save to database
        conn = sqlite3.connect('circuitfix.db')
        c = conn.cursor()
        c.execute("INSERT INTO analyses (user_id, image_path, defects) VALUES (?, ?, ?)",
                 (session['user_id'], filename, json.dumps(detections)))
        analysis_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'image_url': f'/static/uploads/{filename}',
            'annotated_image_url': f'/static/uploads/{os.path.basename(annotated_image_path)}' if annotated_image_path else None,
            'detections': detections
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/report/<int:analysis_id>')
@login_required
def download_report(analysis_id):
    # Get analysis data including image path
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    c.execute("SELECT defects, image_path FROM analyses WHERE id = ? AND user_id = ?", 
              (analysis_id, session['user_id']))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return "Analysis not found", 404
    
    analysis_data = {
        'detections': json.loads(result[0])
    }
    
    # Get the annotated image path (prefer annotated version)
    original_image = result[1]
    annotated_image_path = None
    if original_image:
        # Try to find the annotated version first
        base_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
        annotated_path = base_path.replace('.jpg', '_annotated.jpg').replace('.png', '_annotated.png')
        if os.path.exists(annotated_path):
            annotated_image_path = annotated_path
        elif os.path.exists(base_path):
            annotated_image_path = base_path
    
    # Generate PDF with image
    pdf_data = create_pdf_report(analysis_data, session['username'], annotated_image_path)
    
    return send_file(
        io.BytesIO(pdf_data),
        download_name=f'circuitfix_report_{analysis_id}.pdf',
        as_attachment=True,
        mimetype='application/pdf'
    )

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.')
    return redirect('/')

@app.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    """View detailed analysis with repair guides"""
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    c.execute('''SELECT id, image_path, defects, created_at 
                 FROM analyses WHERE id = ? AND user_id = ?''',
              (analysis_id, session['user_id']))
    result = c.fetchone()
    conn.close()
    
    if not result:
        flash('Analysis not found.')
        return redirect('/dashboard')
    
    detections = json.loads(result[2])
    
    # Count critical defects
    critical_count = sum(1 for d in detections 
                        if d.get('repair_advice', {}).get('severity') == 'Critical')
    
    # Get image URLs
    image_url = f'/static/uploads/{result[1]}'
    annotated_path = result[1].replace('.jpg', '_annotated.jpg').replace('.png', '_annotated.png')
    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], annotated_path)):
        annotated_image_url = f'/static/uploads/{annotated_path}'
    else:
        annotated_image_url = image_url
    
    analysis = {
        'id': result[0],
        'image_path': result[1],
        'created_at': result[3]
    }
    
    return render_template('analysis.html',
                         analysis=analysis,
                         detections=detections,
                         critical_count=critical_count,
                         image_url=image_url,
                         annotated_image_url=annotated_image_url)

@app.route('/knowledge')
@login_required
def knowledge_base():
    """Searchable knowledge base of all defect types"""
    return render_template('knowledge.html', defects=REPAIR_KNOWLEDGE)

@app.route('/analytics')
@login_required
def analytics_page():
    """Analytics dashboard with charts"""
    return render_template('analytics.html')

@app.route('/api/analytics')
@login_required
def get_analytics_data():
    """Get detailed analytics data for charts"""
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    
    # Get all user's analyses
    c.execute('SELECT defects, created_at FROM analyses WHERE user_id = ?',
              (session['user_id'],))
    analyses = c.fetchall()
    conn.close()
    
    # Defect frequency
    defect_counts = {}
    severity_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    daily_counts = {}
    
    for analysis in analyses:
        try:
            defects = json.loads(analysis[0])
            date_str = analysis[1][:10] if analysis[1] else 'Unknown'
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
            
            for defect in defects:
                defect_type = defect.get('defect', 'Unknown')
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                
                severity = defect.get('repair_advice', {}).get('severity', 'Unknown')
                if severity in severity_counts:
                    severity_counts[severity] += 1
        except:
            pass
    
    return jsonify({
        'defect_frequency': defect_counts,
        'severity_distribution': severity_counts,
        'daily_trend': daily_counts,
        'total_analyses': len(analyses)
    })

@app.route('/profile')
@login_required
def profile_page():
    """User profile and settings"""
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    c.execute('SELECT username, email, created_at FROM users WHERE id = ?',
              (session['user_id'],))
    user = c.fetchone()
    c.execute('SELECT COUNT(*) FROM analyses WHERE user_id = ?',
              (session['user_id'],))
    total_analyses = c.fetchone()[0]
    conn.close()
    
    return render_template('profile.html',
                         user={'username': user[0], 'email': user[1], 'created_at': user[2]},
                         total_analyses=total_analyses)

@app.route('/profile/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    
    if new_password != confirm_password:
        flash('New passwords do not match.')
        return redirect('/profile')
    
    if len(new_password) < 8:
        flash('Password must be at least 8 characters long.')
        return redirect('/profile')
    
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE id = ?', (session['user_id'],))
    stored_password = c.fetchone()[0]
    
    if not check_password_hash(stored_password, current_password):
        conn.close()
        flash('Current password is incorrect.')
        return redirect('/profile')
    
    new_hash = generate_password_hash(new_password)
    c.execute('UPDATE users SET password = ? WHERE id = ?',
              (new_hash, session['user_id']))
    conn.commit()
    conn.close()
    
    flash('Password changed successfully!')
    return redirect('/profile')

# ==================== FORUM ROUTES ====================

@app.route('/forum')
@login_required
def forum():
    """Discussion forum main page"""
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    
    # Get all posts with author info and reply count
    c.execute('''
        SELECT fp.id, fp.title, fp.category, fp.views, fp.created_at, 
               u.username, 
               (SELECT COUNT(*) FROM forum_replies WHERE post_id = fp.id) as reply_count
        FROM forum_posts fp
        JOIN users u ON fp.user_id = u.id
        ORDER BY fp.created_at DESC
    ''')
    posts = c.fetchall()
    conn.close()
    
    return render_template('forum.html', posts=posts)

@app.route('/forum/new', methods=['GET', 'POST'])
@login_required
def new_post():
    """Create new forum post"""
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        category = request.form.get('category', 'General')
        
        conn = sqlite3.connect('circuitfix.db')
        c = conn.cursor()
        c.execute('INSERT INTO forum_posts (user_id, title, content, category) VALUES (?, ?, ?, ?)',
                 (session['user_id'], title, content, category))
        post_id = c.lastrowid
        conn.commit()
        conn.close()
        
        flash('Your post has been published!')
        return redirect(f'/forum/{post_id}')
    
    return render_template('forum_new.html')

@app.route('/forum/<int:post_id>')
@login_required
def view_post(post_id):
    """View single forum post with replies"""
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    
    # Increment view count
    c.execute('UPDATE forum_posts SET views = views + 1 WHERE id = ?', (post_id,))
    conn.commit()
    
    # Get post with author
    c.execute('''
        SELECT fp.id, fp.title, fp.content, fp.category, fp.views, fp.created_at, u.username
        FROM forum_posts fp
        JOIN users u ON fp.user_id = u.id
        WHERE fp.id = ?
    ''', (post_id,))
    post = c.fetchone()
    
    if not post:
        conn.close()
        flash('Post not found.')
        return redirect('/forum')
    
    # Get replies
    c.execute('''
        SELECT fr.id, fr.content, fr.created_at, u.username
        FROM forum_replies fr
        JOIN users u ON fr.user_id = u.id
        WHERE fr.post_id = ?
        ORDER BY fr.created_at ASC
    ''', (post_id,))
    replies = c.fetchall()
    conn.close()
    
    return render_template('forum_thread.html', post=post, replies=replies)

@app.route('/forum/<int:post_id>/reply', methods=['POST'])
@login_required
def reply_post(post_id):
    """Add reply to forum post"""
    content = request.form['content']
    
    conn = sqlite3.connect('circuitfix.db')
    c = conn.cursor()
    c.execute('INSERT INTO forum_replies (post_id, user_id, content) VALUES (?, ?, ?)',
             (post_id, session['user_id'], content))
    conn.commit()
    conn.close()
    
    flash('Your reply has been posted!')
    return redirect(f'/forum/{post_id}')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)