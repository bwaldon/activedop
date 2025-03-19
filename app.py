"""Web interface for treebank annotation.

TODO (nschneid):
- validate on accept without entering edit mode
- EVENTUALLY: additional metadata fields e.g. comments, original untokenized sentence

Design notes:

- settings.cfg: The sentences, grammar, user accounts, and other fixed
  configuration parameters are read from a file called "settings.cfg".
  NB: besides the grammar, the treebank the grammar was based on is
  required. Make sure that paths in the grammar parameter file are valid
  (including headrules).
- <filename>.rankings.json: The order in which sentences are annotated
  (prioritization) is fixed before the web service is started, and created by a
  separate command (initpriorities) and stored in a JSON file read at startup
  of the web app.
- annotate.db, schema.sql: Annotations are stored in an sqlite database;
  initialize with "flask initdb".
- session cookie: login status and the per-sentence user interactions are
  stored in a cookie.
- Parsing is done in subprocesses, one for each user, in order to isolate the
  parsing from the Flask process and from other users. Since these subprocesses
  are stored in a global variable (WORKERS), there should only be a single
  Flask process, but multiple threads are needed to get actual parallelism:
  $ flask run --with-threads
  The use of a global variable is a compromise to avoid the complexity of
  running a task queue like celery, or a separate webserver for each user.
"""
import os
import re
import sys
import csv
import json
import sqlite3
import psycopg2
import logging
import subprocess
import click
from math import log
from time import time
from datetime import datetime
from functools import wraps
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, CancelledError
from concurrent.futures.process import BrokenProcessPool
from urllib.parse import urlparse, urlencode, urljoin, unquote
from flask import (Flask, Markup, Response, jsonify, request, session, g, flash, abort,
		redirect, url_for, render_template, send_file, send_from_directory,
		stream_with_context)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from discodop.tree import (Tree, DrawTree, DrawDependencies, ParentedTree,
		writediscbrackettree)
from discodop.treebank import writetree, writedependencies, exporttree
from discodop.treetransforms import canonicalize
from discodop.treebanktransforms import reversetransform
from discodop.parser import probstr
from discodop.disambiguation import testconstraints
from discodop.heads import applyheadrules
from discodop.eval import editdistance
import requests
import base64
from dotenv import load_dotenv
import worker
from workerattr import workerattr
from activedoptree import ActivedopTree, LABELRE, is_punct_label, cgel_to_ptree
from gh_helpers import get_installation_access_token, parse_github_url, check_push_access
sys.path.append('./cgel')
from threading import Lock
try:
	import cgel
	from tree2tex import trees2tex
except ImportError:
	cgel = None
	load_as_cgel = None


app = Flask(__name__)  # pylint: disable=invalid-name
# WORKERS = {}  # dict mapping username to process pool
SENTENCES = None
QUEUE = None
ANNOTATIONHELP = """
- If altering the tokenization, ensure that tokens are numbered sequentially.
- Use _. as the token for a gap.
"""
(NBEST, CONSTRAINTS, DECTREE, REATTACH, RELABEL, REPARSE, EDITDIST, TIME
		) = range(8)
# e.g., "NN-SB/Nom" => ('NN', '-SB', '/Nom')

# Load default config and override config from an environment variable
app.config.update(
		DATABASE=os.path.join(app.root_path, 'annotate.db'),
		SECRET_KEY=None,  # set in settings.cfg to protect session cookies
		DEBUG=False,  # whether to enable Flask debug UI
		LIMIT=100,  # maximum sentence length
		FUNCTIONTAGWHITELIST=(),  # optional list of function tags to always accept
		GRAMMAR=None,  # path to a directory with the initial grammar.
		SENTENCES=None,  # a filename with sentences to annotate, one per line.
		ACCOUNTS=None,  # dictionary mapping usernames to passwords
		ANNOTATIONHELP=None,  # plain text file summarizing the annotation scheme
		CGELVALIDATE=None,  # whether to run the CGEL validator when editing
		PROJ_POS={},  # dictionary mapping POS tags to their corresponding projection tags (for the 'newproj' action)
		PUNCT_TAGS={}, # dictionary mapping idiosyncratic punctuation POS tags to their corresponding tokens
		SYMBOL_TAG=None, # pos tag for symbols (and symbol sequences) that don't have an idiosyncratic tag (in PUNCT_TAGS)
		AMBIG_SYM={}, # 'ambiguous' symbols that can be punctuation or something else depending on context
		INITIAL_PUNCT_LABELS={}, # pos-function labels for punctuation tokens associated with their following non-punctuation tokens
		WORKERS={},  # dict mapping username to process pool
		ACTIVE_PARSES={},  # Track active parse requests by username
		ACTIVE_PARSES_LOCK=Lock(),  # Lock for thread-safe access
		)
app.config.from_pyfile('settings.cfg', silent=True)
app.config.from_envvar('FLASK_SETTINGS', silent=True)
WORKERS = app.config['WORKERS']

app.secret_key = os.getenv("SECRET_KEY")

# GitHub App credentials
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
GITHUB_APP_NAME = os.getenv("GITHUB_APP_NAME", "Tree-Editor")
with open('annotation-editor.2025-03-03.private-key.pem', 'r') as key_file:
	GITHUB_APP_PRIVATE_KEY = key_file.read()
GITHUB_APP_WEBHOOK_SECRET = os.getenv("GITHUB_APP_WEBHOOK_SECRET")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

if app.config['DATABASE'] == 'remote':
	DB_BLANK = '%s'
else:	
	DB_BLANK = '?'

@app.route('/github_login')
def github_login():
	"""
	Redirect to GitHub for user authorization
	"""
	# Generate a random state value for security
	import random
	import string
	state = ''.join(random.choices(string.ascii_lowercase + string.digits, k=20))
	session['oauth_state'] = state
	
	# Store the original URL to redirect back after authentication if any
	if request.args.get('next'):
		session['redirect_after_login'] = request.args.get('next')
	
	# Redirect to GitHub OAuth flow
	return redirect(
		f"https://github.com/login/oauth/authorize"
		f"?client_id={GITHUB_CLIENT_ID}"
		f"&redirect_uri={url_for('github_callback', _external=True)}"
		f"&state={state}"
	)

@app.route('/github_logout')
def github_logout():
	"""
	Log out the user by clearing the user access token
	"""
	session.pop('user_access_token', None)
	session.pop('user_login', None)
	flash("Logged out of GitHub", "success")
	return redirect(url_for('annotate'))

@app.route('/github_callback')
def github_callback():
	"""
	Handle GitHub OAuth callback and obtain user access token
	"""
	# Verify state to prevent CSRF
	if request.args.get('state') != session.get('oauth_state'):
		flash("State verification failed. Please try again.", "error")
		return redirect(url_for('annotate'))
	
	# Exchange code for user access token
	code = request.args.get('code')
	
	if not code:
		flash("Authentication failed - no code received", "error")
		return redirect(url_for('annotate'))
	
	# Request user access token from GitHub
	response = requests.post(
		'https://github.com/login/oauth/access_token',
		headers={'Accept': 'application/json'},
		data={
			'client_id': GITHUB_CLIENT_ID,
			'client_secret': GITHUB_CLIENT_SECRET,
			'code': code,
			'redirect_uri': url_for('github_callback', _external=True)
		}
	)
	
	if response.status_code != 200:
		flash(f"Failed to authenticate: {response.content}", "error")
		return redirect(url_for('annotate'))
	
	data = response.json()
	
	# Store the user access token in session
	session['user_access_token'] = data.get('access_token')
	
	# Get user information
	user_response = requests.get(
		'https://api.github.com/user',
		headers={
			'Authorization': f"token {session['user_access_token']}",
			'Accept': 'application/vnd.github.v3+json'
		}
	)
	
	if user_response.status_code != 200:
		flash("Failed to fetch user information", "error")
		return redirect(url_for('annotate'))
	
	user_data = user_response.json()
	session['user_login'] = user_data.get('login')
	
	# Redirect to the original URL if stored
	if 'redirect_after_login' in session:
		redirect_url = session.pop('redirect_after_login')
		return redirect(redirect_url)
	
	flash(f"Logged in as {user_data.get('login')}", "success")
	return redirect(url_for('annotate'))

@app.route('/from-git/<path:github_url>')
def from_git(github_url):
	"""
	Parse GitHub URL and display file editor
	"""
	# Check if user is logged in - if not, redirect to login with return URL
	if 'user_access_token' not in session:
		return redirect(url_for('github_login', next=request.url))
	
	# Check if the installation ID is present in the session
	if 'installation_id' not in session:
		# Store the URL to redirect back after installation is selected
		session['redirect_after_install'] = url_for('from_git', github_url=github_url)
		return redirect(url_for('select_installation'))
	
	# Normalize URL
	if not github_url.startswith('https://'):
		github_url = 'https://' + github_url
	
	try:
		owner, repo_name, branch, file_path = parse_github_url(github_url)
		
		# Get installation token for the repository
		installation_token = get_installation_access_token(session['installation_id'])
		
		# Get file content from GitHub using the installation token
		# First get the file metadata (including sha)
		content_url = f'https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}?ref={branch}'
		headers = {
			'Authorization': f"token {installation_token}",
			'Accept': 'application/vnd.github.v3+json'
		}
		
		content_response = requests.get(content_url, headers=headers)
		
		if content_response.status_code != 200:
			flash(f"Failed to fetch file metadata: {content_response.status_code}", "error")
			return redirect(url_for('annotate'))
		
		file_data = content_response.json()
		file_content = base64.b64decode(file_data['content']).decode('utf-8')
		
		# Store file info in session
		session['file_info'] = {
			'owner': owner,
			'repo': repo_name,
			'branch': branch,
			'path': file_path,
			'sha': file_data['sha'],
			'original_content': file_content
		}
		
		# Check if user and installation have write permissions to the repository
		has_push_access = check_push_access(session)
		
		# construct treeobj from file content, which has the following format:
		# ```
		# # sent_id = [id]
		# # text = [text]
		# # sent = [tokenized text]
		# # tree_by = [username]
		# [tree]
		# ```
		# where `[tree]` is the tree in CGEL format, starting on the fifth line
		sent_id, text, sent, tree_by, treestr = file_content.split('\n', 4)
		sent_id = sent_id.split('=')[1].strip()
		text = text.split('=')[1].strip()
		sent = sent.split('=')[1].strip()
		tree_by = tree_by.split('=')[1].strip()
		if len(treestr) > 0:
			treeobj = ActivedopTree.from_str(treestr)
			senttok = treeobj.senttok
			sentno = 1
			username = session['username']
			lineno = 1
			rows = max(5, treeobj.treestr().count('\n') + 1)
			msg = ""
			id = generate_id()
			SENTENCES.insert(0, " ".join(senttok))
			refreshqueue(username)
			return render_template('edittree.html',
				prevlink=('/annotate/annotate/%d' % (sentno - 1))
					if sentno > 1 else '/annotate/annotate/%d' % (len(SENTENCES)),
				nextlink=('/annotate/annotate/%d' % (sentno + 1))
					if sentno < len(SENTENCES) else '/annotate/annotate/1',
				unextlink=('/annotate/annotate/%d' % firstunannotated(username))
					if sentno < len(SENTENCES) else '#',
				treestr=treeobj.treestr(), senttok=' '.join(senttok), id=id,
				sentno=sentno, lineno=lineno + 1, totalsents=len(SENTENCES),
				numannotated=numannotated(username),
				poslabels=sorted(t for t in workerattr('poslabels') if ('@' not in t) and (t not in app.config['PUNCT_TAGS']) and (t not in app.config['PUNCT_TAGS'].values()) and (t != app.config['SYMBOL_TAG'])),
				phrasallabels=sorted(t for t in workerattr('phrasallabels') if '}' not in t),
				functiontags=sorted(t for t in (workerattr('functiontags')
					| set(app.config['FUNCTIONTAGWHITELIST'])) if '}' not in t and '@' not in t and t != "p"),
				morphtags=sorted(workerattr('morphtags')),
				annotationhelp=ANNOTATIONHELP,
				rows=rows, cols=100,
				has_push_access=has_push_access,
				file_path=file_path,
				owner=owner,
				repo_name=repo_name,
				source_branch=branch,
				tree_by=tree_by,
				msg=msg)
		else:
			# to create a tokenized sentence, take the 'text' variable, lowercase the first word, split on whitespace, and split off punctuation
			senttok = text.lower().split()
			senttok = [re.split(r'([^\w\s])', token) for token in senttok]
			senttok = [tok for sublist in senttok for tok in sublist if tok]
			senttok = " ".join(senttok)
			id = generate_id()
			SENTENCES.insert(0, senttok)
			refreshqueue(session['username'])

			return redirect(url_for('annotate', sentno=1, github_url=github_url))	
	
	except Exception as e:
		flash(f"Error processing GitHub URL: {str(e)}", "error")
		return redirect(url_for('annotate'))

@app.route('/select-installation')
def select_installation():
	"""
	Let the user select a GitHub App installation
	"""
	if 'user_access_token' not in session:
		return redirect(url_for('github_login', next=request.url))
		
	# Use user access token to get installations
	user_headers = {
		'Authorization': f"Bearer {session['user_access_token']}",
		'Accept': 'application/vnd.github.v3+json'
	}
	
	installations_url = "https://api.github.com/user/installations"
	installations_response = requests.get(installations_url, headers=user_headers)
	
	installations_data = installations_response.json()
	installations = installations_data.get('installations', [])
	
	# For each installation, get the list of repositories
	for installation in installations:
		repos_url = installation.get('repositories_url')
		installation_token = get_installation_access_token(installation['id'])
		
		repos_headers = {
			'Authorization': f"token {installation_token}",
			'Accept': 'application/vnd.github.v3+json'
		}
		
		repos_response = requests.get(repos_url, headers=repos_headers)
		
		if repos_response.status_code == 200:
			repos_data = repos_response.json()
			installation['repositories'] = repos_data.get('repositories', [])
		else:
			installation['repositories'] = []
	
	return render_template('select_installation.html', installations=installations)

@app.route('/set-installation/<int:installation_id>')
def set_installation(installation_id):
	"""
	Set the selected installation ID in the session
	"""
	session['installation_id'] = installation_id
	
	# Redirect to the originally requested URL if it exists
	if 'redirect_after_install' in session:
		redirect_url = session.pop('redirect_after_install')
		return redirect(redirect_url)
	
	return redirect(url_for('annotate'))

@app.route('/save', methods=['POST'])
def save_changes():
	"""
	Save changes to GitHub by creating a new branch or updating an existing one
	"""
	if 'user_access_token' not in session:
		return jsonify({
			'status': 'error',
			'message': "You must be logged in to save changes"
		}), 401
		
	if 'installation_id' not in session:
		return jsonify({
			'status': 'error',
			'message': "No GitHub App installation selected"
		}), 400
		
	# Make file_info optional
	file_info = session.get('file_info', {})
	edited_content = request.json.get('content', '')
	file_path = request.json.get('file_path', '')
	source_branch = request.json.get('source_branch', 'main')
	owner = request.json.get('owner', '')
	repo_name = request.json.get('repo_name', '')

	if not file_path:
		return jsonify({
			'status': 'error',
			'message': "File path is required"
		}), 400
	
	try:
		treeobj = ActivedopTree.from_str(edited_content)

		senttok = [terminal.text if terminal.constituent != 'GAP' else '--' for terminal in treeobj.cgel_tree.terminals()]
		sent = ["# sent = " + " ".join(senttok)]
		# Get original content from file_info if available, otherwise use empty string
		original_content = file_info.get('original_content', '')
		
		# sent_id is the line from the original file that starts with '# sentid ='
		sent_id = [line for line in original_content.split('\n') if line.startswith('# sent_id =')] if original_content else []
		# text_orig is the line from the original file that starts with '# text ='
		text_orig = [line for line in original_content.split('\n') if line.startswith('# text =')] if original_content else []
		# tree_by is the line from the original file that starts with '# tree_by ='
		tree_by = ['# tree_by = ' + request.json.get('tree_by', '[FIELD MISSING]')]

		metadata_lines = sent_id + text_orig + sent + tree_by
		edited_content = "\n".join(metadata_lines) + "\n" + edited_content
		
		branch_name = request.json.get('branch_name', '').strip()
		commit_message = request.json.get('commit_message', '').strip()

		if not commit_message or len(commit_message) == 0:
			return jsonify({
				'status': 'error',
				'message': "Commit message is required"
			}), 400

		if not branch_name or len(branch_name) == 0:
			return jsonify({
				'status': 'error',
				'message': "Branch name is required"
			}), 400
		
		# Get installation token
		installation_token = get_installation_access_token(session['installation_id'])
		
		# Headers for GitHub API requests
		headers = {
			'Authorization': f"token {installation_token}",
			'Accept': 'application/vnd.github.v3+json'
		}
				
		# Check if the app installation has push access
		has_push_access = check_push_access(session)
		
		if not has_push_access:
			return jsonify({
				'status': 'error',
				'message': "This GitHub App installation doesn't have write permissions for this repository"
			}), 403
		
		# Get the reference to the default branch
		ref_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/heads/{source_branch}"
		ref_response = requests.get(ref_url, headers=headers)
		
		if ref_response.status_code != 200:
			return jsonify({
				'status': 'error',
				'message': f"Failed to get branch reference: {ref_response.status_code}"
			}), 500
		
		ref_data = ref_response.json()
		source_sha = ref_data.get('object', {}).get('sha')
		
		overwrite = request.json.get('overwrite', False)
		file_sha = None  # We'll get the file SHA dynamically instead of using file_info
		
		# Only try to create a new branch if not overwriting an existing one
		if not overwrite:
			# Create a new branch - use owner and repo_name from request
			new_ref_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/refs"
			new_ref_data = {
				'ref': f"refs/heads/{branch_name}",
				'sha': source_sha
			}
			
			new_ref_response = requests.post(new_ref_url, headers=headers, json=new_ref_data)
			
			if new_ref_response.status_code == 422:  # Branch already exists
				# Send a confirmation message back to the user
				return jsonify({
					'status': 'error',
					'message': f"Branch '{branch_name}' already exists. Do you want to push this commit to it? (Note: because, we aren't creating a new branch, the 'Source Branch' field will not be used).",
					'branch_exists': True
				}), 422
			
			elif new_ref_response.status_code != 201:
				return jsonify({
					'status': 'error',
					'message': f"Failed to create new branch: {new_ref_response.status_code}"
				}), 500
		else:
			pass  # We'll handle getting file SHA in the next step
			
		# Whether overwriting or not, we need to check if the file exists to get its SHA
		file_contents_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}"
		
		# If overwriting, check in the specified branch
		if overwrite:
			file_contents_url += f"?ref={branch_name}"
		else:
			# Otherwise check in the default branch
			file_contents_url += f"?ref={source_branch}"
			
		file_response = requests.get(file_contents_url, headers=headers)
		
		if file_response.status_code == 200:
			# File exists, get its SHA
			file_data = file_response.json()
			file_sha = file_data['sha']
		
		# Update the file with new content - use owner and repo_name from request
		update_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}"
		
		# Encode the file content as base64
		content_bytes = edited_content.encode('utf-8')
		base64_content = base64.b64encode(content_bytes).decode('utf-8')
		
		# Prepare update data
		update_data = {
			'message': commit_message,
			'content': base64_content,
			'branch': branch_name
		}
		
		# Only include sha if we have one (required for updating existing files)
		if file_sha:
			update_data['sha'] = file_sha
		
		update_response = requests.put(update_url, headers=headers, json=update_data)
		
		if update_response.status_code != 200 and update_response.status_code != 201:
			return jsonify({
				'status': 'error',
				'message': f"Failed to update file: {update_response.status_code}"
			}), 500
		
		# Generate the branch URL to return - use owner and repo_name from request
		branch_url = f"https://github.com/{owner}/{repo_name}/tree/{branch_name}"
		
		# Return success with branch URL
		return jsonify({
			'status': 'success',
			'message': f"Changes saved to branch '{branch_name}'",
			'branch_url': branch_url
		})
		
	except Exception as e:
		return jsonify({
			'status': 'error',
			'message': f"Error saving changes: {str(e)}"
		}), 500
	
def generate_id():
	import random
	import string
	id = None
	# verify that the ID is unique 
	db = getdb()
	cur = db.cursor()
	cur.execute(
		'SELECT id FROM entries ORDER BY sentno ASC'
	)
	entries = cur.fetchall()
	existing_ids = {entry[0] for entry in entries} | {entry[3] for entry in QUEUE}
	while id is None or id in existing_ids:
		id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
	return id

def refreshqueue(username):
	""""Ensures that user can view annotations of sentences not in the 'initpriorities' queue.
	These sentences are shown first, before the prioritized queue."""
	cmd = f'SELECT id, sentno, cgel_tree FROM entries WHERE username = {DB_BLANK} ORDER BY sentno ASC'
	db = getdb()
	cur = db.cursor()
	cur.execute(cmd, (username,))
	dbentries = cur.fetchall()
	queue_ids = [entry[3] for entry in QUEUE]
	for row in dbentries:
		id = row[0]
		sentno = row[1]
		cgel_tree = row[2]
		sent = " ".join(ActivedopTree.from_str(cgel_tree).senttok)
		if id not in queue_ids:
			SENTENCES.insert(0, sent)
			QUEUE.insert(0, [sentno, 0, sent, id])
		# re-index the queue
		for i, entry in enumerate(QUEUE):
			entry[0] = i

@app.cli.command('initpriorities')
@click.option('--username', default='JoeAnnotator', help='Username to initialize priorities for.')
def initpriorities(username):
	"""Order sentences by entropy of their parse trees probabilities.
	Sentences with saved annotations are included first in the order and are not re-parsed."""
	sentfilename = app.config['SENTENCES']
	db = getdb()
	cmd = f'SELECT id FROM entries WHERE username = {DB_BLANK} ORDER BY sentno ASC'
	cur = db.cursor()
	cur.execute(cmd, (username,))
	dbentries = cur.fetchall()
	dbentryids = {a[0] for a in dbentries}
	if sentfilename is None:
		raise ValueError('SENTENCES not configured')
	sentences = []
	with open(sentfilename, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for row in csv_reader:
			if any(field.strip() for field in row.values()):
				sentences.append(row)
	# NB: here we do not use a subprocess to do the parsing
	worker.loadgrammar(app.config['GRAMMAR'], app.config['LIMIT'])
	queue = []
	already_annotated = []
	for n, entry in enumerate(sentences):
		sent = entry['sentence']
		id = entry['id']
		if id in dbentryids:
			app.logger.info('%d. [already annotated] %s',
					n + 1, sent)
			already_annotated.append((n, 0, sent, id))
		else:
			try:
				senttok, parsetrees, _messages, _elapsed = worker.getparses(sent)
			except ValueError:
				parsetrees = []
				senttok = []
			app.logger.info('%d. [parse trees=%d] %s',
					n + 1, len(parsetrees), sent)
			ent = 0
			if parsetrees:
				probs = [prob for prob, _tree, _treestr, _deriv in parsetrees]
				try:
					ent = entropy(probs)  # / log(len(parsetrees), 2)
				except (ValueError, ZeroDivisionError):
					pass
			queue.append((n, ent, sent, id))
	queue.sort(key=lambda x: x[1], reverse=True)
	queue = already_annotated + queue
	rankingfilename = '%s.rankings.json' % sentfilename
	with open(rankingfilename, 'w') as rankingfile:
		json.dump(queue, rankingfile, indent=4)


@app.before_first_request
def initapp():
	"""Load sentences, check config."""
	global SENTENCES, QUEUE, ANNOTATIONHELP
	sentfilename = app.config['SENTENCES']
	if sentfilename is None:
		raise ValueError('SENTENCES not configured')
	if app.config['GRAMMAR'] is None:
		raise ValueError('GRAMMAR not configured')
	if app.config['ACCOUNTS'] is None:
		raise ValueError('ACCOUNTS not configured')
	# read sentences to annotate
	sentences = []
	with open(sentfilename, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for row in csv_reader:
			if any(field.strip() for field in row.values()):
				sentences.append(row['sentence'])
	SENTENCES = sentences
	rankingfilename = '%s.rankings.json' % sentfilename
	if (os.path.exists(rankingfilename) and
			os.stat(rankingfilename).st_mtime
			> os.stat(sentfilename).st_mtime):
		with open(rankingfilename) as rankingfile:
			QUEUE = json.load(rankingfile)
	else:
		raise ValueError('no rankings for sentences, or sentences have\n'
				'been modified; run "flask initpriorities"')
	if app.config['ANNOTATIONHELP'] is not None:
		with open(app.config['ANNOTATIONHELP']) as inp:
			ANNOTATIONHELP = inp.read()


# Database functions
@app.cli.command('initdb')
def initdb():
	"""Initializes the database."""
	db = getdb()
	with app.open_resource('schema.sql', mode='r') as inp:
		schema = inp.read()
	if app.config['DATABASE'] == 'remote':
		with db.cursor() as cur:
			for statement in schema.split(';'):
				if statement.strip():
					cur.execute(statement)
	else:
		db.cursor().executescript(schema)
	db.commit()
	app.logger.info('Initialized the database.')


def connectdb():
	"""Connects to the specific database."""
	if app.config['DATABASE'] == 'remote':
		db_params = {
            "host": os.getenv("DB_HOST"),  
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_UNAME"),       # default is often 'postgres'
            "password": os.getenv("DB_PWD"),
            "port": os.getenv("DB_PORT")                 # default PostgreSQL port
        }
		result = psycopg2.connect(**db_params)
	else:
		result = sqlite3.connect(app.config['DATABASE'])
	return result


def getdb():
	"""Opens a new database connection if there is none yet for the
	current application context."""
	if not hasattr(g, 'db'):
		g.db = connectdb()
	return g.db


@app.teardown_appcontext
def closedb(error):
	"""Closes the database again at the end of the request."""
	if hasattr(g, 'db'):
		g.db.close()

@app.route('/annotate/get_data_psv')
def get_data_psv():
	username = session['username']
	cmd = f'SELECT * FROM entries WHERE username = {DB_BLANK} ORDER BY sentno ASC'
	db = getdb()
	cur = db.cursor()
	cur.execute(cmd, (username,))
	rows = cur.fetchall()
	output_dir = "tmp"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)


	csv_file_path = output_dir + '/output.csv'
	with open(csv_file_path, 'w', newline='') as out_file:
		csv_writer = csv.writer(out_file, delimiter='|')
		column_headers = [description[0] for description in cur.description]
		csv_writer.writerow(column_headers)
		for row in rows:
			csv_writer.writerow(row)
	
	return send_file(csv_file_path, as_attachment=True, attachment_filename='data.csv')

def firstunannotated(username):
	"""Return index of first unannotated sentence,
	according to the prioritized order."""
	db = getdb()
	cmd = f'SELECT id FROM entries WHERE username = {DB_BLANK} ORDER BY sentno ASC'
	cur = db.cursor() 
	cur.execute(cmd, (username,))
	entries = {a[0] for a in cur}
	# sentno=prioritized index, lineno=original index
	for sentno, (_, _, _, id) in enumerate(QUEUE, 1):
		if id not in entries:
			return sentno
	return 1


def numannotated(username):
	"""Number of unannotated sentences for an annotator."""
	db = getdb()
	cmd = f'SELECT count(sentno) FROM entries WHERE username = {DB_BLANK}'
	cur = db.cursor()
	cur.execute(cmd, (username,))
	result = cur.fetchone()
	return result[0]


def getannotation(username, id):
	"""Fetch annotation of a single sentence from database."""
	db = getdb()
	selection = 'cgel_tree, nbest' if app.config['CGELVALIDATE'] else 'tree, nbest'
	cmd = f'select {selection} from entries where username = {DB_BLANK} and id = {DB_BLANK}'
	cur = db.cursor()
	cur.execute(cmd, (username, id))
	entry = cur.fetchone()
	return (None, 0) if entry is None else (entry[0], entry[1])


def readannotations(username=None):
	"""Get all annotations, or ones by a given annotator."""
	db = getdb()
	cur = db.cursor()
	if username is None:
		cmd = 'select sentno, tree from entries order by sentno asc'
		cur.execute(cmd)
	else:
		cmd = f'select sentno, tree from entries where username = {DB_BLANK} order by sentno asc'
		cur.execute(cmd, (username,))
	entries = cur.fetchall()
	return OrderedDict(entries)


def addentry(id, sentno, tree, cgel_tree, actions):
	"""Add an annotation to the database."""
	db = getdb()
	if app.config['DATABASE'] == 'remote':
		query = 'INSERT INTO entries VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
		with db.cursor() as cur:
			cur.execute(query, (id, sentno, session['username'], tree, cgel_tree, *actions, datetime.now().strftime('%F %H:%M:%S')))
	else:
		db.execute(
			'insert or replace into entries '
			'values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
			(id, sentno, session['username'], tree, cgel_tree, *actions,
			datetime.now().strftime('%F %H:%M:%S')))
	db.commit()


def loginrequired(f):
	"""Decorator for views that require a login and running worker pool."""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		if 'username' not in session:
			flash('please log in.')
			return redirect(url_for('login', next=request.url))
		elif session['username'] not in WORKERS:
			return redirect(url_for('dologin', next=request.url))
		return f(*args, **kwargs)
	return decorated_function


def is_safe_url(target):
	"""http://flask.pocoo.org/snippets/62/"""
	ref_url = urlparse(request.host_url)
	test_url = urlparse(urljoin(request.host_url, target))
	return (test_url.scheme in ('http', 'https')
			and ref_url.netloc == test_url.netloc)


# View functions
# NB: when using reverse proxy, the base url should match,
# e.g., https://external.com/annotate/... => http://localhost:5000/annotate/...
@app.route('/')
@app.route('/annotate/')
def main():
	"""Redirect to main page."""
	return redirect(url_for('login'))

@app.route('/annotate/get_id', methods=['GET'])
@loginrequired
def get_id():
	"""Generate a unique 6-character hash ID for a direct entry sentence.
	(Serves as a default ID in the direct entry dialogue window.)"""
	id = generate_id()
	return jsonify({'id': id})

@app.route('/annotate/direct_entry', methods=['POST'])
@loginrequired
def direct_entry():
	"""Directly enter a sentence."""
	sent = request.json.get('sent', '').strip()
	sentid = str(request.json.get('id', '')).strip()
	if len(sent.split()) > app.config['LIMIT']:
		return jsonify({'error': 'Sentence too long. (Maximum length: {})'.format(app.config['LIMIT'])})
	if not sent:
		return jsonify({'error': 'Sentence is empty.'})
	elif not sentid:
		return jsonify({'error': 'Sentence ID is empty.'})
	db = getdb()
	cmd = 'SELECT id FROM entries ORDER BY sentno ASC'
	cur = db.cursor()
	cur.execute(cmd)
	entries = cur.fetchall()
	existing_ids = {entry[0] for entry in entries} | {entry[3] for entry in QUEUE}
	if sentid in existing_ids:
		return jsonify({'error': 'Sentence ID already exists in the database or queue.'})
	else:
		SENTENCES.insert(0, sent)
		QUEUE.insert(0, [0, 0, sent, sentid])
	# re-index the queue
	for i, entry in enumerate(QUEUE):
		entry[0] = i
	return jsonify({'redirect_url': url_for('annotate', sentno=1)})

@app.route('/annotate/login', methods=['GET', 'POST'])
def login():
	"""Check authentication."""
	error = None
	if request.method == 'POST':
		username = request.form['username']
		if (username not in app.config['ACCOUNTS']
				or request.form['password']
				!= app.config['ACCOUNTS'][username]):
			error = 'Invalid username/password'
		else:  # authentication valid
			session['username'] = username
			if request.args.get('next'):
				return redirect(url_for(
						'dologin', next=request.args.get('next')))
			return redirect(url_for('dologin'))
	elif 'username' in session:
		if request.args.get('next'):
			return redirect(url_for(
					'dologin', next=request.args.get('next')))
		return redirect(url_for('dologin'))
	return render_template(
			'login.html', error=error, totalsents=len(SENTENCES))

def loadgrammar(username):
	_, lang = os.path.split(os.path.basename(app.config['GRAMMAR']))
	app.logger.info('Loading grammar %r', lang)
	if username in WORKERS and isinstance(WORKERS[username], ProcessPoolExecutor):
		WORKERS[username].shutdown(wait=False)
	pool = ProcessPoolExecutor(max_workers=1)
	future = pool.submit(
			worker.loadgrammar,
			app.config['GRAMMAR'], app.config['LIMIT'])
	future.result()
	app.logger.info('Grammar %r loaded.', lang)
	# train on annotated sentences
	annotations = readannotations()
	if annotations:
		app.logger.info('training on %d previously annotated sentences',
				len(annotations))
		trees, sents = [], []
		headrules = pool.submit(worker.getprop, 'headrules').result()
		for block in annotations.values():
			# HOTFIX for ROOT error
			blocklns = block.splitlines()
			for iln,blockln in enumerate(blocklns):
				if '\tROOT\t' in blockln and '\t0\t' not in blockln:
					blocklns[iln] = blocklns[iln].replace('\tROOT\t', '\tXXX-XXX\t')
			block = '\n'.join(blocklns)

			item = exporttree(block.splitlines())
			canonicalize(item.tree)
			if headrules:
				applyheadrules(item.tree, headrules)
			trees.append(item.tree)
			sents.append(item.sent)
		if False and app.config['DEBUG']:
			future = NoFuture(worker.augment, trees, sents)
		else:
			future = pool.submit(worker.augment, trees, sents)
		future.result()
	WORKERS[username] = pool

@app.route('/annotate/dologin')
def dologin():
	"""Start worker pool and redirect when done."""
	def generate(url):
		yield (
				'<!doctype html>'
				'<title>redirect</title>'
				'You were logged in successfully. ')
		if username in WORKERS and isinstance(WORKERS[username], ProcessPoolExecutor):
			try:
				_ = WORKERS[username].submit(
						worker.getprop, 'headrules').result()
			except BrokenProcessPool:
				pass  # fall through
			else:
				yield "<script>window.location.replace('%s');</script>" % url
				return
		yield 'Loading grammar; this will take a few seconds ...'
		loadgrammar(username)	
		yield "<script>window.location.replace('%s');</script>" % url
	nexturl = request.args.get('next')
	if not is_safe_url(nexturl) or 'username' not in session:
		return abort(400)
	username = session['username']
	return Response(stream_with_context(generate(
			nexturl or url_for('annotate'))))
	# return generate(nexturl or url_for('annotate'))


# FIXME: add automatic session expiration?
@app.route('/annotate/logout')
def logout():
	"""Log out: clear session, shut down worker pool."""
	if 'username' in session and session['username'] in WORKERS:
		pool = WORKERS.pop(session['username'])
		pool.shutdown(wait=False)
	session.pop('username', None)
	flash('You were logged out')
	return redirect(url_for('main'))


@app.route('/annotate/annotate/', defaults={'sentno': -1, 'github_url': None})
@app.route('/annotate/annotate/<int:sentno>', defaults={'github_url': None})
@app.route('/annotate/annotate/<int:sentno>/<path:github_url>')
@loginrequired
def annotate(sentno, github_url):
	"""Serve the main annotation page for a sentence."""
	username = session['username']
	refreshqueue(username)
	if sentno == -1:
		sentno = firstunannotated(username)
		return redirect(url_for('annotate', sentno=sentno, github_url=github_url))
	session['actions'] = [0, 0, 0, 0, 0, 0, 0, time()]
	lineno = QUEUE[sentno - 1][0]
	id = QUEUE[sentno - 1][3]
	sent = SENTENCES[lineno]
	senttok, _ = worker.postokenize(sent)
	annotation, n = getannotation(username, id)
	if annotation is not None: # a tree is saved in the database
		# go directly to edit mode
		return redirect(url_for(
				'edit', sentno=sentno, annotated=1, n=n, github_url=github_url))
	# render annotate mode: browsing parser outputs for the sentence
	return render_template(
			'annotate.html',
			prevlink=str(sentno - 1) if sentno > 1 else str(len(SENTENCES)),
			nextlink=str(sentno + 1) if sentno < len(SENTENCES) else str(1),
			sentno=sentno, lineno=lineno + 1,
			totalsents=len(SENTENCES),
			numannotated=numannotated(username),
			annotationhelp=ANNOTATIONHELP,
			github_url=github_url,
			sent=' '.join(senttok))	# includes any gaps

@app.route('/undoaccept', methods=['POST'])
def undoaccept():
	sentid = request.json.get('sentid', 0)
	username = session['username']
	cmd = f'DELETE FROM entries WHERE username = {DB_BLANK} AND id = {DB_BLANK}'
	db = getdb()
	cur = db.cursor()
	cur.execute(cmd, (username, sentid))
	db.commit()
	# reload the grammar
	loadgrammar(username)
	return jsonify({"success": True})

@app.route('/retokenize', methods=['POST'])
def retokenize():
	sentno = int(request.json.get('sentno', 0))
	newtext = request.json.get('newtext', 0)
	if len(newtext.split()) > app.config['LIMIT']:
		return jsonify({'success': False,	
			'error': 'Sentence too long. (Maximum length: {})'.format(app.config['LIMIT'])})
	lineno = QUEUE[sentno - 1][0]
	SENTENCES[lineno] = newtext
	return jsonify({"success": True})

@app.route('/reload_grammar', methods=['GET'])
@loginrequired
def reload_grammar():
	username = session['username']
	# terminate the worker pool
	WORKERS[username].shutdown(wait=False)
	del WORKERS[username]
	_, lang = os.path.split(os.path.basename(app.config['GRAMMAR']))
	app.logger.info('Loading grammar %r', lang)
	pool = ProcessPoolExecutor(max_workers=1)
	future = pool.submit(
			worker.loadgrammar,
			app.config['GRAMMAR'], app.config['LIMIT'])
	future.result()
	app.logger.info('Grammar %r loaded.', lang)
	# train on annotated sentences
	annotations = readannotations()
	if annotations:
		app.logger.info('training on %d previously annotated sentences',
				len(annotations))
		trees, sents = [], []
		headrules = pool.submit(worker.getprop, 'headrules').result()
		for block in annotations.values():
			# HOTFIX for ROOT error
			blocklns = block.splitlines()
			for iln,blockln in enumerate(blocklns):
				if '\tROOT\t' in blockln and '\t0\t' not in blockln:
					blocklns[iln] = blocklns[iln].replace('\tROOT\t', '\tXXX-XXX\t')
			block = '\n'.join(blocklns)

			item = exporttree(block.splitlines())
			canonicalize(item.tree)
			if headrules:
				applyheadrules(item.tree, headrules)
			trees.append(item.tree)
			sents.append(item.sent)
			future = pool.submit(worker.augment, trees, sents)
		future.result()
	WORKERS[username] = pool
	return jsonify({"success": True})

@app.route('/annotate/cancel_parse', methods=['POST'])
def cancel_parse():
	"""Cancel any running parse requests for the current user."""
	username = session.get('username')
	if not username:
		return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
	
	with app.config['ACTIVE_PARSES_LOCK']:
		active_futures = app.config['ACTIVE_PARSES'].pop(username, [])
		for future in active_futures:
			if not future.done():
				future.cancel()
	
	return jsonify({'status': 'success'})

# Modify the parse route to track active requests
@app.route('/annotate/parse', methods=['POST'])
@loginrequired
def parse():
	"""Display parse. To be invoked by an AJAX call."""
	sentno = int(request.json.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	github_url = request.json.get('github_url', None)
	username = session['username']
	require = request.json.get('require', '')
	block = request.json.get('block', '')
	urlprm = dict(sentno=sentno, github_url=github_url)
	if require and require != '':
		urlprm['require'] = require
	if block and block != '':
		urlprm['block'] = block
	require, block = parseconstraints(require, block)
	if require or block:
		session['actions'][CONSTRAINTS] += 1
		session.modified = True
	if len(sent.split()) > app.config['LIMIT']:
		return jsonify({'error': 'Sentence too long. (Maximum length: {})'.format(app.config['LIMIT'])})
	
	# Submit the parse task to the worker pool
	future = WORKERS[username].submit(
			worker.getparses,
			sent, require, block)
	
	# Track this future in the active parses for this user
	with app.config['ACTIVE_PARSES_LOCK']:
		if username not in app.config['ACTIVE_PARSES']:
			app.config['ACTIVE_PARSES'][username] = []
		app.config['ACTIVE_PARSES'][username].append(future)
	
	try:
		# Get the result of the future

		resp = future.result()
		
		# Remove the future from active parses
		with app.config['ACTIVE_PARSES_LOCK']:
			if username in app.config['ACTIVE_PARSES'] and future in app.config['ACTIVE_PARSES'][username]:
				app.config['ACTIVE_PARSES'][username].remove(future)
		
		senttok, parsetrees, messages, elapsed = resp
		maxdepth = ''
	
		if not parsetrees:
			result = ('no parse! reload page to clear constraints, '
					'or continue with next sentence.')
			nbest = dep = depsvg = ''
		else:
			dep = depsvg = ''
			if workerattr('headrules'):
				dep = writedependencies(parsetrees[0][1], senttok, 'conll')
				depsvg = Markup(DrawDependencies.fromconll(dep).svg())
			result = ''
			dectree, maxdepth, _ = decisiontree(parsetrees, senttok, urlprm)
			prob, ptree, _treestr, _fragments = parsetrees[0]
			treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
			nbest = Markup('%s\nbest tree: %s' % (
					dectree,
					('%(n)d. [%(prob)s] '
					'<a href="/annotate/accept?%(urlprm)s">accept this tree</a>; '
					'<a href="/annotate/edit?%(urlprm)s">edit</a>; '
					'<a href="/annotate/deriv?%(urlprm)s">derivation</a>\n\n'
					'%(tree)s'
					% dict(
						n=1,
						prob=probstr(prob),
						urlprm=urlencode(dict(urlprm, n=1)),
						tree=treeobj.gtree()))))
		msg = '\n'.join(messages)
		elapsed = 'CPU time elapsed: %s => %gs' % (
				' '.join('%gs' % a for a in elapsed), sum(elapsed))
		info = '\n'.join((
				'length: %d;' % len(senttok), msg, elapsed,
				'most probable parse trees:',
				''.join('%d. [%s] %s' % (n + 1, probstr(prob),
						writediscbrackettree(treestr, senttok))
						for n, (prob, _tree, treestr, _deriv)
						in enumerate(parsetrees)
						if treestr is not None)
				+ '\n'))
	except CancelledError:
			# If cancelled, return a minimal result
			return jsonify({'error': 'Parse request cancelled'})
	finally:
		# Always remove the future from active parses
		with app.config['ACTIVE_PARSES_LOCK']:
			if username in app.config['ACTIVE_PARSES'] and future in app.config['ACTIVE_PARSES'][username]:
				app.config['ACTIVE_PARSES'][username].remove(future)
	return render_template('annotatetree.html', sent=sent, result=result,
		nbest=nbest, info=info, dep=dep, depsvg=depsvg, maxdepth=maxdepth,
		msg='%d parse trees' % len(parsetrees))

@app.route('/annotate/filter')
@loginrequired
def filterparsetrees():
	"""For a parse tree in the cache, return a filtered set of its n-best
	parses matching current constraints."""
	username = session['username']
	session['actions'][CONSTRAINTS] += 1
	session.modified = True
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	urlprm = dict(sentno=sentno)
	require = request.args.get('require', '')
	block = request.args.get('block', '')
	if require and require != '':
		urlprm['require'] = require
	if block and block != '':
		urlprm['block'] = block
	require, block = parseconstraints(require, block)
	frequire = request.args.get('frequire', '')
	fblock = request.args.get('fblock', '')
	frequire, fblock = parseconstraints(frequire, fblock)
	resp = WORKERS[username].submit(
			worker.getparses,
			sent, require, block).result()
	senttok, parsetrees, _messages, _elapsed = resp
	parsetrees_ = [(n, prob, ActivedopTree(ptree = ptree, senttok = senttok), treestr, frags)
			for n, (prob, ptree, treestr, frags) in enumerate(parsetrees)
			if treestr is None or testconstraints(treestr, frequire, fblock)]
	if len(parsetrees_) == 0:
		return ('No parse trees after filtering; try pressing Re-parse, '
				'or reload page to clear constraints.\n')
	nbest = Markup('%d parse trees\n%s' % (
			len(parsetrees_),
			'\n'.join('%(n)d. [%(prob)s] '
				'<a href="/annotate/accept?%(urlprm)s">accept this tree</a>; '
				'<a href="/annotate/edit?%(urlprm)s">edit</a>; '
				'<a href="/annotate/deriv?%(urlprm)s">derivation</a>\n\n'
				'%(tree)s' % dict(
					n=n + 1,
					prob=probstr(prob),
					urlprm=urlencode(dict(urlprm, n=n + 1)),
					# ad_tree: ActivedopTree object
					tree=ad_tree.gtree())
				for n, prob, ad_tree, _treestr, fragments in parsetrees_)))
	return nbest


@app.route('/annotate/deriv')
@loginrequired
def showderiv():
	"""Render derivation for a given parse tree in cache."""
	username = session['username']
	n = int(request.args.get('n'))  # 1-indexed
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	require = request.args.get('require', '')
	block = request.args.get('block', '')
	require, block = parseconstraints(require, block)
	resp = WORKERS[username].submit(
			worker.getparses,
			sent, require, block).result()
	senttok, parsetrees, _messages, _elapsed = resp
	_prob, tree, _treestr, fragments = parsetrees[n - 1]
	return Markup(
			'<pre>Fragments used in the highest ranked derivation'
			' of this parse tree:\n%s\n%s</pre>' % (
			'\n\n'.join(
				'%s\n%s' % (w, DrawTree(frag).text(unicodelines=True, html=True))
				for frag, w in fragments or ()),
			DrawTree(tree, senttok).text(
				unicodelines=True, html=True, funcsep='-', maxwidth=30)))


@app.route('/annotate/edit')
@loginrequired
def edit():
	"""Edit tree manually."""
	sentno = int(request.args.get('sentno'))  # 1-indexed
	github_url = request.args.get('github_url', None)
	file_path = ''
	owner = ''
	repo_name = ''
	branch = ''
	orig_content = session.get('file_info', {}).get('original_content', None)
	# tree_by is the line in orig_content that starts with '# tree_by ='
	if orig_content:
		tree_by = [line for line in orig_content.split('\n') if line.startswith('# tree_by =')]
		if tree_by:
			tree_by = tree_by[0].split('=')[1].strip()
	else:
		tree_by = ''

	# process the URL
	if github_url and github_url != 'None':
		github_url = unquote(github_url)
		try:
			owner, repo_name, branch, file_path = parse_github_url(github_url)
		except ValueError as e:
			flash(f"Error: {str(e)}", "error")
			return redirect(url_for('annotate'))
	
	lineno = QUEUE[sentno - 1][0]
	id = QUEUE[sentno - 1][3]
	sent = SENTENCES[lineno]
	senttok, _ = worker.postokenize(sent)
	username = session['username']
	if 'dec' in request.args:
		session['actions'][DECTREE] += int(request.args.get('dec', 0))
	session.modified = True
	msg = ''
	if request.args.get('annotated') == '1': # there is a saved tree
		msg = Markup('<font color=red>You have already annotated '
				'this sentence.</font><button id="undo" onclick="undoAccept()">Revert (deletes annotation)</button>')
		id = QUEUE[sentno - 1][3]
		treestr, n = getannotation(username, id) # get tree from database
		treeobj = ActivedopTree.from_str(treestr)
		senttok = treeobj.senttok
		# ensures that SENTENCES array is updated with the tokenized sentence
		SENTENCES[lineno] = ' '.join(senttok)
	elif 'n' in request.args: # edit the nth automatic parse
		# msg = Markup('<button id="undo" onclick="goback()">Go back</button>')
		n = int(request.args.get('n', 1))
		session['actions'][NBEST] = n
		require = request.args.get('require', '')
		block = request.args.get('block', '')
		require, block = parseconstraints(require, block)
		resp = WORKERS[username].submit(
				worker.getparses,
				sent, require, block).result()
		senttok, parsetrees, _messages, _elapsed = resp
		ptree = parsetrees[n - 1][1]
		treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
	else:
		return 'ERROR: pass n or tree argument.'
	rows = max(5, treeobj.treestr().count('\n') + 1)
	return render_template('edittree.html',
			prevlink=('/annotate/annotate/%d' % (sentno - 1))
				if sentno > 1 else '/annotate/annotate/%d' % (len(SENTENCES)),
			nextlink=('/annotate/annotate/%d' % (sentno + 1))
				if sentno < len(SENTENCES) else '/annotate/annotate/1',
			unextlink=('/annotate/annotate/%d' % firstunannotated(username))
				if sentno < len(SENTENCES) else '#',
			treestr=treeobj.treestr(), senttok=' '.join(senttok), id=id,
			sentno=sentno, lineno=lineno + 1, totalsents=len(SENTENCES),
			numannotated=numannotated(username),
			poslabels=sorted(t for t in workerattr('poslabels') if ('@' not in t) and (t not in app.config['PUNCT_TAGS']) and (t not in app.config['PUNCT_TAGS'].values()) and (t != app.config['SYMBOL_TAG'])),
			phrasallabels=sorted(t for t in workerattr('phrasallabels') if '}' not in t),
			functiontags=sorted(t for t in (workerattr('functiontags')
				| set(app.config['FUNCTIONTAGWHITELIST'])) if '}' not in t and '@' not in t and t != "p"),
			morphtags=sorted(workerattr('morphtags')),
			file_path=file_path,
			tree_by=tree_by,
			owner = owner,
			repo_name = repo_name,
			source_branch=branch,
			annotationhelp=ANNOTATIONHELP,
			rows=rows, cols=100,
			msg=msg)

@app.route('/annotate/redraw', methods=['POST'])
@loginrequired
def redraw():
	"""Validate and re-draw tree."""
	data = request.get_json()
	sentno = int(data.get('sentno')) # 1-indexed
	has_error = False
	link = ('''<a href="#" onclick="accept()">accept this tree</a>
		<input type="hidden" id="sentno" value="%d">'''
	% (sentno))
	try:
		treeobj = ActivedopTree.from_str(data.get('tree'))
		msg = treeobj.validate()
	except ValueError as err:
		msg = str(err)
		treeobj = None
		has_error = True
		return jsonify({'msg': msg,
				  'accept_link': link,
				  'gtree': '',
				  'has_error': has_error})
	tree_to_accept = treeobj.treestr()
	tree_for_editdist = re.sub(r'\s+', ' ', str(tree_to_accept))
	oldtree = request.args.get('oldtree', '')
	oldtree = re.sub(r'\s+', ' ', oldtree)
	if oldtree and tree_for_editdist != oldtree:
		session['actions'][EDITDIST] += editdistance(tree_for_editdist, oldtree)
		session.modified = True
	return jsonify({'msg': msg,
				  'accept_link': link,
				  'gtree': treeobj.gtree(add_editable_attr=True),
				  'has_error': has_error})

def graphical_operation_preamble(treestr):
	treeobj = ActivedopTree.from_str(treestr)
	if app.config['CGELVALIDATE'] is None:
		cgel_tree_terminals = None
	else:
		cgel_tree_terminals = treeobj.cgel_tree.terminals(gaps=True)
	return treeobj, cgel_tree_terminals

def graphical_operation_postamble(dt, senttok, cgel_tree_terminals, sentno):
	ptree = ParentedTree.convert(canonicalize(dt.nodes[0]))
	treeobj = ActivedopTree(ptree = ptree, senttok = senttok, 
						 cgel_tree_terminals = cgel_tree_terminals)
	msg = treeobj.validate()
	link = ('<a href="/annotate/accept?%s">accept this tree</a>'
		% urlencode(dict(sentno=sentno, tree=treeobj.treestr())))	
	return treeobj, link, msg

@app.route('/annotate/newlabel', methods=['POST'])
@loginrequired
def newlabel():
	"""Re-draw tree with newly picked label."""
	data = request.get_json()
	treestr = data.get('tree')
	try:
		treeobj, cgel_tree_terminals = graphical_operation_preamble(treestr)
	except ValueError as err:
		return Markup(str(err))
	senttok = treeobj.senttok
	# FIXME: re-factor; check label AFTER replacing it
	# now actually replace label at nodeid
	_treeid, nodeid = data.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	dt = DrawTree(treeobj.ptree, treeobj.senttok)
	m = LABELRE.match(dt.nodes[nodeid].label)
	try:
		if data.get('label') is not None:
			label = data.get('label', '')
			dt.nodes[nodeid].label = (label
					+ (m.group(2) or '')
					+ (m.group(3) or ''))
		elif data.get('function') is not None:
			label = data.get('function', '')
			if label == '':
				dt.nodes[nodeid].label = '%s%s' % (
						m.group(1), m.group(3) or '')
			else:
				dt.nodes[nodeid].label = '%s-%s%s' % (
						m.group(1), label, m.group(3) or '')
		elif data.get('morph') is not None:
			label = data.get('morph', '')
			if label == '':
				dt.nodes[nodeid].label = '%s%s' % (
						m.group(1), m.group(2) or '')
			else:
				dt.nodes[nodeid].label = '%s%s/%s' % (
						m.group(1), m.group(2) or '', label)
		else:
			raise ValueError('expected label or function argument')
		treeobj, link, msg = graphical_operation_postamble(dt, senttok, cgel_tree_terminals, int(data.get('sentno'))) 
		session['actions'][RELABEL] += 1
		session.modified = True
		return jsonify({'msg': msg,
					'accept_link': link,
					'error' : '',
					'gtree': treeobj.gtree(add_editable_attr=True),
					'treestr': treeobj.treestr()})
	except ValueError as err:
		error = str(err)
		return jsonify({'msg': msg,
					'accept_link': link,
					'error' : error,
					'gtree': treeobj.gtree(add_editable_attr=True),
					'treestr': treeobj.treestr()})

@app.route('/annotate/reattach', methods=['POST'])
@loginrequired
def reattach():
	"""Re-draw tree after re-attaching node under new parent."""
	data = request.get_json()
	treestr = data.get('tree')
	try:
		treeobj, cgel_tree_terminals = graphical_operation_preamble(treestr)
	except ValueError as err:
		return Markup(str(err))
	# kludge (can't deep copy treeobj)
	old_treeobj, _ = graphical_operation_preamble(treestr)
	try:
		dt = DrawTree(treeobj.ptree, treeobj.senttok)
		senttok = treeobj.senttok
		if data.get('newparent') == 'deletenode':
			# remove nodeid by replacing it with its children
			_treeid, nodeid = data.get('nodeid', '').lstrip('t').split('_')
			nodeid = int(nodeid)
			x = dt.nodes[nodeid]
			if nodeid == 0 or isinstance(x[0], int):
				raise ValueError('cannot remove ROOT or POS node')
			else:
				children = list(x)
				x[:] = []
				for y in dt.nodes[0].subtrees():
					if any(child is x for child in y):
						i = y.index(x)
						y[i:i + 1] = children
						tree = canonicalize(dt.nodes[0])
						dt = DrawTree(tree, senttok)  # kludge..
						break
		elif data.get('nodeid', '') == 'newproj':
			# splice in a new node under parentid
			_treeid, newparent = data.get('newparent', ''
					).lstrip('t').split('_')
			newparent = int(newparent)
			y = dt.nodes[newparent]
			children = list(y)
			label = y.label
			new_top_label = label
			if isinstance(y[0], int):
				old_pos = LABELRE.match(label).group(1)
				new_pos = app.config['PROJ_POS'].get(old_pos, 'Clause')
				new_top_label = new_pos + "-Head"
			y[:] = [Tree(label, children)]
			y.label = new_top_label
			tree = canonicalize(dt.nodes[0])
			dt = DrawTree(tree, senttok)  # kludge..
		elif data.get('nodeid', '').startswith('newlabel_'):
			# splice in a new node under parentid
			_treeid, newparent = data.get('newparent', ''
					).lstrip('t').split('_')
			newparent = int(newparent)
			label = data.get('nodeid').split('_', 1)[1]
			y = dt.nodes[newparent]
			if isinstance(y[0], int):
				raise ValueError('cannot add node under POS tag')
			else:
				children = list(y)
				y[:] = []
				y[:] = [Tree(label, children)]
				tree = canonicalize(dt.nodes[0])
				dt = DrawTree(tree, senttok)  # kludge..
		else:  # re-attach existing node at existing new parent
			_treeid, nodeid = data.get('nodeid', '').lstrip('t').split('_')
			nodeid = int(nodeid)
			_treeid, newparent = data.get('newparent', ''
					).lstrip('t').split('_')
			newparent = int(newparent)
			# remove node from old parent
			# dt.nodes[nodeid].parent.pop(dt.nodes[nodeid].parent_index)
			x = dt.nodes[nodeid]
			y = dt.nodes[newparent]

			def find_self_and_sisters(tree, subtree):
				parent = None
				sisters = []

				# Helper function to find the parent of the subtree
				def find_parent(node, target):
					nonlocal parent
					if target in node.children:
						parent = node
						return True
					for child in node.children:
						if isinstance(child, int):
							return False
						elif find_parent(child, target):
							return True
					return False

				# Find the parent of the subtree
				find_parent(tree, subtree)

				if parent:
					# Collect all children of the parent node
					sisters = [child for child in parent.children]

				return sisters
			
			def extract_adjacent_punctuation(arr, target):
				# Find the index of the target character
				try:
					target_index = arr.index(target)
				except ValueError:
					return []  # If target is not in the list, return an empty list
		
				# Initialize the result list with the target character
				result = [target]

				# Collect punctuation characters to the left of the target
				left_index = target_index - 1
				while left_index >= 0 and is_punct_label(arr[left_index].label):
					result.insert(0, arr[left_index])
					left_index -= 1
				
				# Collect punctuation characters to the right of the target
				right_index = target_index + 1
				while right_index < len(arr) and is_punct_label(arr[right_index].label):
					result.append(arr[right_index])
					right_index += 1
		
				return result
			
			for node in x.subtrees():
				if node is y:
					raise ValueError('cannot re-attach subtree'
							' under (descendant of) itself\n')
			else:
				for node in dt.nodes[0].subtrees():
					if any(child is x for child in node):
						if len(node) > 1:
							self_and_sisters = find_self_and_sisters(dt.nodes[0], x)
							self_and_nearbypunct = extract_adjacent_punctuation(self_and_sisters, x)
							for s in self_and_nearbypunct:
								# iteratively move all sister punctuation to the target. 
								# (prevents problematic crossover movement of non-punctuation nodes over punctuation nodes)
								# punctuation positions are subsequently re-canonicalized when ActivedopTree is reconstructed
								node.remove(s)
								dt.nodes[newparent].append(s)
							tree = canonicalize(dt.nodes[0])
							dt = DrawTree(tree, senttok)  # kludge..
						else:
							raise ValueError('re-attaching only child creates'
									' empty node %s; remove manually\n' % node)
						break
		treeobj, link, msg = graphical_operation_postamble(dt, senttok, cgel_tree_terminals, int(data.get('sentno')))
		if treeobj.senttok != old_treeobj.senttok:
			raise ValueError('movement would result in reordered tokens')
		session['actions'][REATTACH] += 1
		session.modified = True
		return jsonify({'msg': msg,
				'accept_link': link,
				'gtree': treeobj.gtree(add_editable_attr=True),
				'error': '',
				'treestr': treeobj.treestr()})
	except Exception as err:
		msg = old_treeobj.validate()
		link = ('<a href="/annotate/accept?%s">accept this tree</a>'
			% urlencode(dict(sentno=int(data.get('sentno')), tree=old_treeobj.treestr())))
		error = "ERROR: " + str(err)
		return jsonify({'msg': msg,
				  'accept_link': link,
				  'gtree': old_treeobj.gtree(add_editable_attr=True),
				  'error': error,
				  'treestr': old_treeobj.treestr()})


@app.route('/annotate/reparsesubtree')
@loginrequired
def reparsesubtree():
	"""Re-parse selected subtree."""
	sentno = int(request.args.get('sentno'))  # 1-indexed
	username = session['username']
	treeobj = ActivedopTree.from_str(request.args.get('tree'))
	dt = DrawTree(treeobj.ptree, treeobj.senttok)
	_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	subseq = sorted(dt.nodes[nodeid].leaves())
	subsent = ' '.join(treeobj.senttok[n] for n in subseq)
	# FIXME only works when root label of tree matches label in grammar.
	# need a single label that works across all stages.
	root = dt.nodes[nodeid].label
	# root = grammar.tolabel[next(iter(grammar.tblabelmapping[root]))]
	resp = WORKERS[username].submit(
			worker.getparses,
			subsent,
			(), (),
			root=root).result()
	_senttok, parsetrees, _messages, _elapsed = resp
	app.logger.info('%d-%d. [parse trees=%d] %s',
			sentno, nodeid, len(parsetrees), subsent)
	print(parsetrees[0][1])
	nbest = Markup('<pre>%d parse trees\n'
			'<a href="javascript: toggle(\'nbest\'); ">cancel</a>\n'
			'%s</pre>' % (
			len(parsetrees),
			'\n'.join('%(n)d. [%(prob)s] '
				'<a href="#" onClick="picksubtree(%(n)d); ">'
				'use this subtree</a>; '
				'\n\n'
				'%(tree)s' % dict(
					n=n + 1,
					prob=probstr(prob),
					tree=DrawTree(tree, subsent.split()).text(
						unicodelines=True, html=True, funcsep='-',
						morphsep='/', nodeprops='t%d' % (n + 1), maxwidth=30))
				for n, (prob, tree, _treestr, fragments)
				in enumerate(parsetrees))))
	return nbest


@app.route('/annotate/replacesubtree')
@loginrequired
def replacesubtree():
	n = int(request.args.get('n', 0))
	sentno = int(request.args.get('sentno'))  # 1-indexed
	username = session['username']
	try:
		treeobj = ActivedopTree.from_str(request.args.get('tree'))
	except ValueError as err:
		return str(err)
	error = ''
	dt = DrawTree(treeobj.ptree, treeobj.senttok)
	cgel_tree_terminals = treeobj.cgel_tree.terminals(gaps=True)
	_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	subseq = sorted(dt.nodes[nodeid].leaves())
	subsent = ' '.join(treeobj.senttok[n] for n in subseq)
	root = dt.nodes[nodeid].label
	resp = WORKERS[username].submit(
			worker.getparses,
			subsent, (), (),
			root=root).result()
	_senttok, parsetrees, _messages, _elapsed = resp
	newsubtree = parsetrees[n - 1][1]
	pos = sorted(list(newsubtree.subtrees(
			lambda n: isinstance(n[0], int))),
			key=lambda n: n[0])
	for n, a in enumerate(pos):
		a[0] = subseq[n]
	dt.nodes[nodeid][:] = newsubtree[:]
	ptree = ParentedTree.convert(canonicalize(dt.nodes[0]))
	treeobj = ActivedopTree(ptree = ptree, senttok = treeobj.senttok, 
						 cgel_tree_terminals = cgel_tree_terminals)
	msg = treeobj.validate()
	session['actions'][REPARSE] += 1
	session.modified = True
	link = ('<a href="/annotate/accept?%s">accept this tree</a>'
			% urlencode(dict(sentno=sentno, tree=treeobj.treestr())))
	return jsonify({'msg': msg,
				  'accept_link': link,
				  'gtree': treeobj.gtree(add_editable_attr=True),
				  'error': error,
				  'treestr': treeobj.treestr()})


@app.route('/annotate/accept', methods=['GET', 'POST'])
@loginrequired
def accept():
	"""Store parse & redirect to next sentence."""
	if request.method == 'POST':
		# request.get_json() returns a dictionary
		data = request.get_json()
	elif request.method == 'GET':
		data = request.args
	# should include n referring to which n-best tree is to be accepted,
	# or tree in discbracket format if tree was manually edited.
	sentno = int(data.get('sentno'))  # 1-indexed
	lineno = QUEUE[sentno - 1][0]
	id = QUEUE[sentno - 1][3]
	sent = SENTENCES[lineno]
	username = session['username']
	actions = session['actions']
	actions[TIME] = int(round(time() - actions[TIME]))
	treestr = None
	if 'dec' in data:
		actions[DECTREE] += int(data.get('dec', 0))
	if 'tree' in data:
		n = 0
		treeobj = ActivedopTree.from_str(data.get('tree'))
		tree_to_train = treeobj.ptree
		senttok = treeobj.senttok
		cgel_tree = treeobj.cgel_tree
		# the tokenization may have been updated with gaps, so store the new one
		SENTENCES[lineno] = ' '.join(senttok)
		if False:
			reversetransform(tree, senttok, ('APPEND-FUNC', 'addCase'))
	else:
		n = int(data.get('n', 0))
		require = data.get('require', '')
		block = data.get('block', '')
		require, block = parseconstraints(require, block)
		resp = WORKERS[username].submit(
				worker.getparses,
				sent, require, block).result()
		senttok, parsetrees, _messages, _elapsed = resp
		ptree = parsetrees[n - 1][1]
		treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
		tree_to_train, cgel_tree = treeobj.ptree, treeobj.cgel_tree
		if False:
			# strip function tags
			for node in tree.subtrees():
				node.label = LABELRE.match(node.label).group(1)
	actions[NBEST] = n
	session.modified = True
	block = writetree(tree_to_train.copy(deep=True), senttok, str(lineno + 1), 'export',
		comment='%s %r' % (username, actions))
	app.logger.info(block)
	treeout = block
	addentry(id, lineno, treeout, str(cgel_tree), actions)	# save annotation in the database
	WORKERS[username].submit(worker.augment, [tree_to_train], [senttok])	# update the parser's grammar
	# validate and stay on this sentence if there are issues
	if treestr:
		msg = treeobj.validate()
		if 'ERROR' in msg or 'WARNING' in msg:
			flash('Your annotation for sentence %d was stored %r but may contain errors. Please click Validate to check.' % (sentno, actions))
			if request.method == 'POST':
				return jsonify({'redirect': url_for('annotate', sentno=sentno)})
			else:
				return redirect(url_for('annotate', sentno=sentno))
	flash('Your annotation for sentence %d was stored %r' % (sentno, actions))
	if request.method == 'POST':
		return jsonify({'redirect': url_for('annotate', sentno=1) if sentno >= len(SENTENCES) else url_for('annotate', sentno=sentno+1)})
	else:
		return (redirect(url_for('annotate', sentno=1))
			if sentno >= len(SENTENCES)
			else redirect(url_for('annotate', sentno=sentno+1)))

@app.route('/annotate/context/<int:lineno>')
def context(lineno):
	"""Show all sentences, in original order."""
	ranking = {a[0]: n for n, a in enumerate(QUEUE, 1)}
	return render_template('context.html',
			sentences=SENTENCES, ranking=ranking, lineno=lineno)


@app.route('/annotate/export')
def export():
	"""Export annotations by current user."""
	return Response(
			''.join(readannotations(session['username']).values()),
			mimetype='text/plain')

@app.route('/annotate/download_pdf')
def download_pdf():
	cgeltree = cgel.parse(request.args.get('tree'))
	cgel_latex = trees2tex(cgeltree)
	output_dir = "tmp"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(os.path.join(output_dir, "file.tex"), 'w') as latex_file:
		latex_file.write(cgel_latex)

	subprocess.run(['pdflatex', '-output-directory', output_dir, os.path.join(output_dir, "file.tex")])

	pdf_path = os.path.join(output_dir, "file.pdf")

	return send_file(pdf_path, as_attachment=True, attachment_filename='downloaded_file.pdf')

@app.route('/annotate/exportcgeltree')
def exportcgeltree():
	"""Produce single tree in .cgel format"""
	cgeltree = request.args.get('tree')
	return Response(cgeltree, mimetype='text/plain')

@app.route('/annotate/favicon.ico')
@app.route('/favicon.ico')
def favicon():
	"""Serve the favicon."""
	return send_from_directory(os.path.join(app.root_path, 'static'),
			'parse.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/annotate/static/script.js')
def javascript():
	"""Serve javascript."""
	return send_from_directory(os.path.join(app.root_path, 'static'),
			'script.js', mimetype='text/javascript')


@app.route('/annotate/static/style.css')
def stylecss():
	"""Serve style.css."""
	return send_from_directory(os.path.join(app.root_path, 'static'),
			'style.css', mimetype='text/css')

def entropy(seq):
	"""Calculate entropy of a probability distribution.

	>>> entropy([0.25, 0.25, 0.25, 0.25])  # high uncertainty, high entropy
	2.0
	>>> entropy([0.9, 0.05, 0.05])  # low uncertainty, low entropy
	0.5689955935892812
	"""
	if not seq:
		return 0
	probmass = sum(seq)
	probs = [prob / probmass for prob in seq]
	return -sum(p * log(p, 2) for p in probs)


def parseconstraints(require, block):
	"""
	>>> parseconstraints("NP 0-2\tPP 0-1,4", "")
	(('NP', [0, 1, 2]), ('PP', [0, 1, 4])), ()
	"""
	def constr(item):
		"""Parse a single constraint."""
		label, span = item.split(' ', 1)
		seq = []
		for rng in span.split(','):
			if '-' in rng:
				b, c = rng.split('-')
				seq.extend(range(int(b), int(c) + 1))
			else:
				seq.append(int(rng))
		return label, seq

	if require:
		require = tuple((label, tuple(indices))
				for label, indices in sorted(map(constr, require.split('\t'))))
	else:
		require = ()
	if block:
		block = tuple((label, tuple(indices))
				for label, indices in sorted(map(constr, block.split('\t'))))
	else:
		block = ()
	return require, block


def getspans(tree):
	"""Yield spans of Tree object."""
	for node in tree.subtrees():
		if node is not tree:  # skip root
			yield node.label, tuple(sorted(node.leaves()))


def decisiontree(parsetrees, sent, urlprm):
	"""Create a decision tree to select among n trees."""
	# The class labels are the n-best trees 0..n
	# The attributes are the labeled spans in the trees; they split the n-best
	# trees into two sets with and without that span.
	spans = {}
	if len(parsetrees) <= 1:
		return '', 0, None
	for n, (_prob, tree, _, _) in enumerate(parsetrees):
		for span in getspans(tree):
			# simplest strategy: store presence of span as binary feature
			# perhaps better: use weight from tree probability
			spans.setdefault(span, set()).add(n)

	# create decision tree with scikit-learn
	features = list(spans)
	featurenames = ['[%s %s]' % (label, ' '.join(sent[n] for n in leaves))
			for label, leaves in features]
	data = np.array([[n in spans[span] for span in features]
			for n in range(len(parsetrees))], dtype=bool)
	estimator = DecisionTreeClassifier(random_state=0)
	estimator.fit(data, range(len(parsetrees)),
			sample_weight=[prob for prob, _, _, _ in parsetrees])
	path = estimator.decision_path(data)

	def rec(tree, n=0, depth=0):
		"""Recursively produce a string representation of a decision tree."""
		if tree.children_left[n] == tree.children_right[n]:
			x = tree.value[n].nonzero()[1][0]
			prob, _tree, _treestr, _fragments = parsetrees[x]
			thistree = ('%(n)d. [%(prob)s] '
					'<a href="/annotate/accept?%(urlprm)s">accept this tree</a>; '
					'<a href="/annotate/edit?%(urlprm)s">edit</a>; '
					'<a href="/annotate/deriv?%(urlprm)s">derivation</a><br>\n\n'
					% dict(
						n=x + 1,
						prob=probstr(prob),
						urlprm=urlencode(dict(urlprm, n=x + 1, dec=depth))))
			return ('<span id="d%d" style="display: none; ">%stree %d:\n'
					'%s</span>' % (n, depth * '\t', x + 1, thistree))
		left = tree.children_left[n]
		right = tree.children_right[n]
		return ('<span id=d%(n)d style="display: %(display)s; ">'
				'%(indent)s%(constituent)s '
				'<a href="javascript: showhide(\'d%(right)s\', \'d%(left)s\', '
					'\'dd%(exright)s\', \'%(numtrees)s\'); ">'
					'good constituent</a> '
				'<a href="javascript: showhide(\'d%(left)s\', \'d%(right)s\', '
					'\'dd%(exleft)s\', \'%(numtrees)s\'); ">'
					'bad constituent</a><br>'
				'%(subtree1)s%(subtree2)s</span>' % dict(
				n=n,
				display='block' if n == 0 else 'none',
				indent=depth * 4 * ' ',
				constituent=featurenames[tree.feature[n]],
				left=left, right=right,
				exleft=path[:, left].nonzero()[0][0],
				exright=path[:, right].nonzero()[0][0],
				numtrees=len(parsetrees),
				subtree1=rec(tree, left, depth + 1),
				subtree2=rec(tree, right, depth + 1),
				))
	nodes = rec(estimator.tree_)
	leaves = []
	seen = set()
	for n in range(estimator.tree_.node_count):
		x = estimator.tree_.value[n].nonzero()[1][0]
		if x in seen:
			continue
		seen.add(x)
		_prob, xtree, _treestr, _fragments = parsetrees[x]
		thistree = DrawTree(xtree, sent).text(
				unicodelines=True, html=True, funcsep='-', morphsep='/',
				nodeprops='t%d' % (x + 1), maxwidth=30)
		leaves.append('<span id="dd%d" style="display: none; ">%s</span>' %
				(x, thistree))
	return nodes + ''.join(leaves), estimator.tree_.max_depth, path

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)

@app.cli.command('cgel2export')
@click.option('--inputfile')
@click.option('--outputfile')
def cgel2export(inputfile, outputfile):
	"""Convert a list of cgel trees from inputfile to Negra export format; write to outputfile
	Produces a list of Negra corpus export-format trees that can be used to train the parser."""
	import copy
	result = []
	with open (inputfile, 'r') as f:
		key = 0
		for tree in cgel.trees(f):
			ptree, senttok = cgel_to_ptree(tree)
			print(tree.metadata)
			treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
			# remove -p function label for training the parser
			for subt in treeobj.ptree.subtrees(lambda t: t.height() == 2):
				if subt.label.endswith("-p"):
					subt.label = subt.label[:-2]
			ptree = treeobj.ptree.copy(deep=True)
			block = writetree(ptree, treeobj.senttok, key=str(key), fmt='export')
			result.append(block)
			key += 1
	with open(outputfile, 'w') as f:
		f.write(''.join(result))
