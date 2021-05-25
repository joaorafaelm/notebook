---
layout: post
title: GraphQL and Django in 5 minutes
categories: [graphql, django]
comments: true
toc: false
---

> **TL;DR** Jump to the [coding part](#models-and-graphql-schema) or get the code [here](https://github.com/joaorafaelm/graphql-django-example).

# What is GraphQL?
GraphQL query is a string that is sent to a server to be interpreted and fulfilled, which then returns JSON back to the client.
It was created by Facebook in 2012 and the first [specification draft](http://facebook.github.io/graphql/) was made public in 2015.

In this tutorial I will cover the basics of working with GraphQL and Django.

# Getting started
Before creating the project and all, make sure you have [virtualenv](https://virtualenv.pypa.io/en/stable/) installed, so that the packages used in this tutorial won't be installed system-wide.

```bash
# Clone the repo
git clone https://github.com/joaorafaelm/graphql-django-example;
cd graphql-django-example;

# Create virtualenv
virtualenv venv && source venv/bin/activate;

# Install django and graphene
pip install -r requirements.txt;

# Setup db
python manage.py migrate;
```

Run `python manage.py loaddata books.json` to populate the db, or run `python manage.py createsuperuser` and then add some data using the admin interface.

# Models and GraphQL Schema
This example is going to use the models Author, Book and Publisher.
```python
# bookstore/store/models.py
from django.db import models

class Publisher(models.Model):
    name = models.CharField(max_length=30)
    website = models.URLField()

    def __str__(self):
        return self.name

class Author(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=40)
    email = models.EmailField()

    def __str__(self):
        return '%s %s' % (self.first_name, self.last_name)

class Book(models.Model):
    title = models.CharField(max_length=100)
    authors = models.ManyToManyField(Author)
    publisher = models.ForeignKey(Publisher)
    publication_date = models.DateField()

    def __str__(self):
        return self.title
```

After creating the models, the [Schema](http://graphql.org/learn/schema/#type-language) should be created, which will be used to serve the API.

At the time of writing this post, [graphene-django](https://github.com/graphql-python/graphene-django) version is 1.3 and it does not handle ManyToMany fields properly, that is why the `resolve_authors` method was added. This issue has been [resolved](https://github.com/graphql-python/graphene-django/issues/155) for the next release.

```python
# bookstore/schema.py
import graphene
from graphene_django.types import DjangoObjectType
from graphene_django.debug import DjangoDebug
from bookstore.store.models import Author, Book, Publisher

class AuthorType(DjangoObjectType):
    class Meta:
        model = Author

class BookType(DjangoObjectType):
    authors = graphene.List(AuthorType)

    # Many To Many fix until next release.
    # https://github.com/graphql-python/graphene-django/issues/155
    @graphene.resolve_only_args
    def resolve_authors(self):
        return self.authors.all()

    class Meta:
        model = Book

class PublisherType(DjangoObjectType):
    class Meta:
        model = Publisher

class Query(graphene.ObjectType):
    all_authors = graphene.List(AuthorType)
    all_books = graphene.List(BookType)
    all_publishers = graphene.List(PublisherType)

    # Debug field (rawSql, parameters etc).
    debug = graphene.Field(DjangoDebug, name='__debug')

    def resolve_all_authors(self, args, context, info):
        return Author.objects.all()

    def resolve_all_books(self, args, context, info):
        return Book.objects.select_related('publisher').all()

    def resolve_all_publishers(self, args, context, info):
        return Publisher.objects.all()

schema = graphene.Schema(query=Query)
```
Last but not least, the GraphQL URL must be added into the urls.py file.

```python
# bookstore/urls.py
from django.conf.urls import url
from django.contrib import admin
from graphene_django.views import GraphQLView
from bookstore.schema import schema

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^graphql', GraphQLView.as_view(graphiql=True, schema=schema)),
]
```
You can now run `python manage.py runserver` and start using the API at [http://localhost:8000/graphql](http://localhost:8000/graphql).

# Querying and debugging
[GraphiQL](https://github.com/graphql/graphiql) provides a graphical interactive in-browser GraphQL IDE, including some features such as syntax highlighting, real-time error reporting, automatic query completion etc.

I will show some query examples, but you can learn more about querying at [graphql.org/learn/queries/](http://graphql.org/learn/queries/).

Given the following query, we can retrieve all books registered along with their authors.
```javascript
{
  allBooks {
    title,
    authors {
      firstName,
      lastName
    }
  }
}
```
And the response...
```json
{
  "data": {
    "allBooks": [
      {
        "title": "Resurrection",
        "authors": [
          {
            "firstName": "Leo",
            "lastName": "Tolstoy"
          }
        ]
      },
      {
        "title": "Childhood",
        "authors": [
          {
            "firstName": "Leo",
            "lastName": "Tolstoy"
          }
        ]
      }
    ]
  }
}
```

Using the `__debug` field you can get information about the actual SQL query.
```javascript
{
  allAuthors {lastName}
  __debug {
    sql {rawSql, duration}
  }
}
```
Response:
```json
{
  "data": {
    "allAuthors": [
      {
        "lastName": "King"
      },
      {
        "lastName": "Tolstoy"
      },
      {
        "lastName": "Gaiman"
      },
      {
        "lastName": "Pratchett"
      }
    ],
    "__debug": {
      "sql": [
        {
          "rawSql": "SELECT \"store_author\".\"id\", \"store_author\".\"first_name\", \"store_author\".\"last_name\", \"store_author\".\"email\" FROM \"store_author\"",
          "duration": 0.0009260177612304688
        }
      ]
    }
  }
}
```

All this code is on my [Github](https://github.com/joaorafaelm/graphql-django-example/). Please do fork it and make pull requests regarding any issues or improvements you may have with my code.
